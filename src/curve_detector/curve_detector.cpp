#include "curve_detector.hpp"
#include <iostream>

#include <fstream>
#include "recon_config.hpp"

#ifndef CURVE_RECONSTRUCTION_ROOT
#define CURVE_RECONSTRUCTION_ROOT "."
#endif

using namespace cv;

CurveDetector::CurveDetector(int _refine, double _scale, double _sigma_scale, double _quant,
                                           double _ang_th, double _log_eps, double _density_th, int _n_bins)
                                           : img_width(0), img_height(0), LOG_NT(0),
                                             SCALE(_scale), doRefine(_refine), SIGMA_SCALE(_sigma_scale), QUANT(_quant),
                                             ANG_TH(_ang_th), LOG_EPS(_log_eps), DENSITY_TH(_density_th), N_BINS(_n_bins),
                                             edge_det_module_(load_module()){}

pybind11::object CurveDetector::load_module(){
    pybind11::module sys = pybind11::module::import("sys");
    pybind11::list sys_path = sys.attr("path");
    sys_path.append(std::string(CURVE_RECONSTRUCTION_ROOT) + "/TEED_INFER/");
    return pybind11::module::import("TEED_inference");
}

void CurveDetector::detect(const Mat& _image, std::vector<Curve>& _curves_new, std::vector<double>& runtimes)
{
    _curves_new.clear();

    image = _image;
    assert(!image.empty() && image.type() == CV_8UC1);

    const double prec = CV_PI * ANG_TH / 180;
    const double p = ANG_TH / 180;
    const double rho = QUANT / sin(prec);

    TicToc tt;
    if(SCALE != 1)
    {
        Mat gaussian_img;
        const double sigma = (SCALE < 1)?(SIGMA_SCALE / SCALE):(SIGMA_SCALE);
        const double sprec = 3;
        const unsigned int h =  (unsigned int)(ceil(sigma * sqrt(2 * sprec * log(10.0))));
        Size ksize(1 + 2 * h, 1 + 2 * h);
        GaussianBlur(image, gaussian_img, ksize, sigma);
        resize(gaussian_img, scaled_image, Size(), SCALE, SCALE, INTER_LINEAR_EXACT);
    }
    else
    {
        scaled_image = image;
    }
    ll_angle(rho, N_BINS);
    LOG_NT = 5 * (log10(double(img_width)) + log10(double(img_height))) / 2 + log10(11.0);
    MIN_REG_SIZE = size_t(-LOG_NT/log10(p));

    double ll_angle_runtime = tt.toc();

#if SHOW_REGION_GROW
    cv::cvtColor(angle_image_.clone(), region_grow_image_, cv::COLOR_GRAY2BGR);
#endif

    double get_region_runtime = 0.0, fitting_runtime = 0.0;
    std::vector<Object> objects;
    used = Mat_<uchar>::zeros(scaled_image.size());
    int feature_cnt = 0;
    for(size_t i = 0; i < ordered_points.size(); ++i)
    {
        const Point2i& point = ordered_points[i].p;

        if((used.at<uchar>(point) == NOTUSED) && (angles.at<double>(point) != NOTDEF))
        {
            std::deque<RegionPoint> reg;
            double reg_angle;
            std::deque<double> local_angles;

            tt.tic();
            bool suc = get_region(point, reg, prec, p, reg_angle, local_angles);
            get_region_runtime += tt.toc();
            if(!suc) continue;

            tt.tic();
            Object obj;
            obj.region = reg;
            obj.reg_angle = reg_angle;

            rect rec;
            region2rect(reg, reg_angle, prec, p, rec);

            double _thre = 7.0;
            obj.is_curve = (rec.width > _thre) ? true : false;

            if(obj.is_curve)
            {
                suc = get_curve(obj);
                if(!suc) continue;
                obj.curve.is_curved = true;
            }
            else
            {
                suc = get_line(obj, point, prec, p);
                if(!suc) continue;

                obj.curve.is_curved = false;
                obj.curve.points.clear();
                cv::Point2d endpoint1(obj.rec.x1, obj.rec.y1), endpoint2(obj.rec.x2, obj.rec.y2);
                double len = cv::norm(endpoint1 - endpoint2);
                int sam_num = static_cast<int>(std::round(len * 2));
                sam_num = std::max(sam_num, 2);
                for(int j = 0; j < sam_num; ++j)
                {
                    double t = double(j) / (sam_num - 1);
                    cv::Point2d pt = endpoint1 * (1.0 - t) + endpoint2 * t;
                    obj.curve.points.emplace_back(pt);
                }
                obj.curve.arc_length = cv::norm(obj.curve.points.front() - obj.curve.points.back());

                obj.curve.pixels.clear();
                obj.curve.pixels.reserve(obj.region.size());

                for(const auto& p : obj.region)
                {

                    obj.curve.pixels.emplace_back(cv::Point(p.x, p.y));
                }
            }
            objects.emplace_back(obj);
            feature_cnt++;
            _curves_new.emplace_back(obj.curve);

            fitting_runtime += tt.toc();

        }
    }
    runtimes.clear();
    runtimes.push_back(ll_angle_runtime);
    runtimes.push_back(get_region_runtime);
    runtimes.push_back(fitting_runtime);
#if VISUALIZATION
    drawObjects_nurbs(objects, "[Detection] Objects");
#endif
}

bool CurveDetector::get_region(const Point2i& s, std::deque<RegionPoint>& reg,
    const double& prec, const double p, double& reg_angle, std::deque<double>& local_angles)
{
#if SHOW_REGION_GROW
    cv::Mat img = region_grow_image_.clone();

#endif

    reg.clear();
    local_angles.clear();
    double angle_front, angle_back;
    std::deque<double> angles_front_buff, angles_back_buff;

    std::deque<double> raw_angles;

    RegionPoint seed;
    seed.x = s.x;
    seed.y = s.y;
    seed.used = &used.at<uchar>(s);
    seed.angle = angles.at<double>(s);
    seed.modgrad = modgrad.at<double>(s);

    reg.emplace_back(seed);
    angles_front_buff.push_back(seed.angle);
    angles_back_buff.push_back(seed.angle);
    raw_angles.push_back(seed.angle);
    local_angles.push_back(seed.angle);
    *seed.used = USED;

#if SHOW_REGION_GROW
    cv::Mat display;
    used.convertTo(display, CV_8UC1, 255);
    cv::imshow("used", display);
    cv::waitKey(1);
    img.at<cv::Vec3b>(seed.y, seed.x) = cv::Vec3b(0, 0, 255);
    cv::Mat img_resized;
    cv::resize(img, img_resized, cv::Size(), 2.0, 2.0, cv::INTER_NEAREST);
    cv::imshow("Region Grow Visualization", img_resized);
    cv::waitKey(0);
#endif

    angle_front = seed.angle;
    angle_back = seed.angle;

    reg_angle = seed.angle;
    float _sumdx = float(std::cos(reg_angle));
    float _sumdy = float(std::sin(reg_angle));

    int seed_index = 0, front_index = 0, back_index = 0;
    while(1)
    {
        bool front_stop = false, back_stop = false;
        if(back_index >= 0 && back_index < reg.size() && !back_stop)
        {
            const RegionPoint& rpoint = reg[back_index];
            int xx_min = std::max(rpoint.x - 1, 0), xx_max = std::min(rpoint.x + 1, img_width - 1);
            int yy_min = std::max(rpoint.y - 1, 0), yy_max = std::min(rpoint.y + 1, img_height - 1);

            for(int yy = yy_min; yy <= yy_max; ++yy)
            {
                uchar* used_row = used.ptr<uchar>(yy);
                const double* angles_row = angles.ptr<double>(yy);
                const double* modgrad_row = modgrad.ptr<double>(yy);
                for(int xx = xx_min; xx <= xx_max; ++xx)
                {
                    uchar& is_used = used_row[xx];
                    if(is_used == USED) continue;

                    cv::Point2f vec1(xx - rpoint.x, yy - rpoint.y),
                                vec2(std::cos(angle_back), std::sin(angle_back));
                    if(std::abs(front_index - seed_index) < 10 && vec1.x * vec2.x + vec1.y * vec2.y < 0) continue;

                    if(isAligned(xx, yy, angle_back, prec))
                    {
                        const double& angle = angles_row[xx];

                        is_used = USED;
                        RegionPoint region_point;
                        region_point.x = xx;
                        region_point.y = yy;
                        region_point.used = &is_used;
                        region_point.modgrad = modgrad_row[xx];
                        region_point.angle = angle;

                        reg.emplace_back(region_point);
#if SHOW_REGION_GROW
                        cv::Mat display;
                        used.convertTo(display, CV_8UC1, 255);
                        cv::imshow("used", display);
                        cv::waitKey(1);
                        img.at<cv::Vec3b>(region_point.y, region_point.x) = cv::Vec3b(0, 255, 0);
                        cv::Mat img_resized;
                        cv::resize(img, img_resized, cv::Size(), 2.0, 2.0, cv::INTER_NEAREST);
                        cv::imshow("Region Grow Visualization", img_resized);
                        cv::waitKey(0);
#endif

                        angles_back_buff.emplace_back(region_point.angle);
                        if(angles_back_buff.size() < WINDOW_SIZE)
                        {
                            int idx = seed_index - 1;
                            while(angles_back_buff.size() < WINDOW_SIZE && idx >= 0)
                            {
                                angles_back_buff.emplace_front(reg[idx].angle);
                                idx--;
                            }
                        }
                        while(angles_back_buff.size() > WINDOW_SIZE)
                            angles_back_buff.pop_front();

                        float sumdx = 0, sumdy = 0;
                        for(const auto& a : angles_back_buff)
                        {
                            sumdx += cos(float(a));
                            sumdy += sin(float(a));
                        }
                        angle_back = fastAtan2(sumdy, sumdx) * DEG_TO_RADS;

                        raw_angles.push_back(region_point.angle);
                        local_angles.push_back(angle_back);

                        _sumdx += cos(float(angle));
                        _sumdy += sin(float(angle));
                        reg_angle = fastAtan2(_sumdy, _sumdx) * DEG_TO_RADS;

                    }
                }
            }
            if(back_index < reg.size() - 1)
                back_index++;
            else
                back_stop = true;
        }

        if(front_index >= 0 && front_index < reg.size() && !front_stop)
        {
            const RegionPoint& rpoint = reg[front_index];
            int xx_min = std::max(rpoint.x - 1, 0), xx_max = std::min(rpoint.x + 1, img_width - 1);
            int yy_min = std::max(rpoint.y - 1, 0), yy_max = std::min(rpoint.y + 1, img_height - 1);
            for(int yy = yy_min; yy <= yy_max; ++yy)
            {
                uchar* used_row = used.ptr<uchar>(yy);
                const double* angles_row = angles.ptr<double>(yy);
                const double* modgrad_row = modgrad.ptr<double>(yy);
                for(int xx = xx_min; xx <= xx_max; ++xx)
                {
                    uchar& is_used = used_row[xx];
                    if(is_used == USED) continue;

                    cv::Point2f vec1(xx - rpoint.x, yy - rpoint.y),
                                vec2(std::cos(angle_front + CV_PI), std::sin(angle_front + CV_PI));
                    if(std::abs(front_index - seed_index) < 10 && vec1.x * vec2.x + vec1.y * vec2.y < 0) continue;

                    if(isAligned(xx, yy, angle_front, prec))
                    {
                        const double& angle = angles_row[xx];

                        is_used = USED;
                        RegionPoint region_point;
                        region_point.x = xx;
                        region_point.y = yy;
                        region_point.used = &is_used;
                        region_point.modgrad = modgrad_row[xx];
                        region_point.angle = angle;

                        reg.emplace_front(region_point);
#if SHOW_REGION_GROW
                        cv::Mat display;
                        used.convertTo(display, CV_8UC1, 255);
                        cv::imshow("used", display);
                        cv::waitKey(1);
                        img.at<cv::Vec3b>(region_point.y, region_point.x) = cv::Vec3b(255, 0, 0);
                        cv::Mat img_resized;
                        cv::resize(img, img_resized, cv::Size(), 2.0, 2.0, cv::INTER_NEAREST);
                        cv::imshow("Region Grow Visualization", img_resized);
                        cv::waitKey(0);
#endif

                        front_index++;
                        back_index++;
                        seed_index++;
                        angles_front_buff.emplace_back(region_point.angle);
                        if(angles_front_buff.size() < WINDOW_SIZE){
                            int idx = seed_index + 1;
                            while(angles_front_buff.size() < WINDOW_SIZE && idx < reg.size())
                            {
                                angles_front_buff.emplace_front(reg[idx].angle);
                                idx++;
                            }
                        }
                        while(angles_front_buff.size() > WINDOW_SIZE)
                            angles_front_buff.pop_front();

                        float sumdx = 0, sumdy = 0;
                        for(const auto& a : angles_front_buff)
                        {
                            sumdx += cos(float(a));
                            sumdy += sin(float(a));
                        }
                        angle_front = fastAtan2(sumdy, sumdx) * DEG_TO_RADS;

                        raw_angles.push_front(region_point.angle);
                        local_angles.push_front(angle_front);

                        _sumdx += cos(float(angle));
                        _sumdy += sin(float(angle));
                        reg_angle = fastAtan2(_sumdy, _sumdx) * DEG_TO_RADS;

                    }
                }
            }
            if(front_index > 0)
                front_index--;
            else
                front_stop = true;
        }
        if(front_stop && back_stop){
            break;
        }
    }

    if(reg.size() - 1 - seed_index < WINDOW_SIZE && seed_index >= WINDOW_SIZE - 1)
    {
        for(int i = seed_index; i < reg.size(); ++i)
        {
            double _angle = 0.0;
            for(int j = i - WINDOW_SIZE + 1; j <= i; ++j)
            {
                _angle += reg[j].angle;
            }
            _angle /= WINDOW_SIZE;

            double _x = reg[i].x, _y = reg[i].y;
            if(!isAligned(_x, _y, _angle, prec))
            {
                for(int k = 0; k < reg.size() - i; ++k)
                {
                    used.ptr<uchar>(reg.back().y)[reg.back().x] = NOTUSED;
#if SHOW_REGION_GROW
                    cv::Mat display;
                    used.convertTo(display, CV_8UC1, 255);
                    cv::imshow("used", display);
                    cv::waitKey(1);
                    img.at<cv::Vec3b>(reg.back().y, reg.back().x) = cv::Vec3b(100, 100, 255);
                    cv::Mat img_resized;
                    cv::resize(img, img_resized, cv::Size(), 2.0, 2.0, cv::INTER_NEAREST);
                    cv::imshow("Region Grow Visualization", img_resized);
                    cv::waitKey(0);
#endif
                    reg.pop_back();
                    local_angles.pop_back();
                }
                break;
            }
        }
    }

    if(seed_index < WINDOW_SIZE && reg.size() - seed_index >= WINDOW_SIZE)
    {
        for(int i = seed_index; i >= 0; --i)
        {
            double _angle = 0.0;
            for(int j = i; j < i + WINDOW_SIZE; ++j)
            {
                _angle += reg[j].angle;
            }
            _angle /= WINDOW_SIZE;

            double _x = reg[i].x, _y = reg[i].y;
            if(!isAligned(_x, _y, _angle, prec))
            {
                for(int k = 0; k <= i; ++k)
                {
                    used.ptr<uchar>(reg.front().y)[reg.front().x] = NOTUSED;
#if SHOW_REGION_GROW
                    cv::Mat display;
                    used.convertTo(display, CV_8UC1, 255);
                    cv::imshow("used", display);
                    cv::waitKey(1);
                    img.at<cv::Vec3b>(reg.front().y, reg.front().x) = region_grow_image_.at<cv::Vec3b>(reg.front().y, reg.front().x);
                    cv::Mat img_resized;
                    cv::resize(img, img_resized, cv::Size(), 2.0, 2.0, cv::INTER_NEAREST);
                    cv::imshow("Region Grow Visualization", img_resized);
                    cv::waitKey(0);
#endif

                    reg.pop_front();
                    local_angles.pop_front();
                }
                break;
            }
        }
    }

#if SHOW_REGION_GROW
    region_grow_image_ = img.clone();
#endif

    if(reg.size() < MIN_REG_SIZE)
    {

        return false;
    }

    return true;
}

bool CurveDetector::get_line(Object& line_obj, const Point2i& seed, const double prec, const double p)
{
    region2rect(line_obj.region, line_obj.reg_angle, prec, p, line_obj.rec);
    double log_nfa = -1;
    if(doRefine > LSD_REFINE_NONE)
    {

        if(!refine(line_obj.region, line_obj.reg_angle, seed, prec, p, line_obj.rec, DENSITY_TH)) { return false; }

        if(doRefine >= LSD_REFINE_ADV)
        {

            log_nfa = rect_improve(line_obj.rec);
            if(log_nfa <= LOG_EPS) { return false; }
        }
    }

    line_obj.rec.x1 += 0.5; line_obj.rec.y1 += 0.5;
    line_obj.rec.x2 += 0.5; line_obj.rec.y2 += 0.5;
    if(SCALE != 1)
    {
        line_obj.rec.x1 /= SCALE; line_obj.rec.y1 /= SCALE;
        line_obj.rec.x2 /= SCALE; line_obj.rec.y2 /= SCALE;
        line_obj.rec.width /= SCALE;
    }

    cv::Point2d endpoint1(line_obj.rec.x1, line_obj.rec.y1), endpoint2(line_obj.rec.x2, line_obj.rec.y2);
    double len = cv::norm(endpoint1 - endpoint2);

    if(len < OBJ_MIN_LEN) return false;

    return true;
}

bool CurveDetector::get_curve(Object& curve_obj)
{

    std::vector<LNLib::XYZ> points;
    points.reserve(curve_obj.region.size());
    for(const auto& p : curve_obj.region)
    {
        points.emplace_back(LNLib::XYZ(p.x, p.y, 0.0));
    }

    LNLib::LN_NurbsCurve curve;
    bool result = LNLib::NurbsCurve::LeastSquaresApproximation(recon_cfg().p_nurbs, points, 10, curve);
    assert(result);

    double rough_length = LNLib::NurbsCurve::ApproximateLength(curve, LNLib::IntegratorType::GaussLegendre);

    double S = std::sqrt(scaled_image.rows * scaled_image.cols);
    int ctrl_point_num = std::max(5, static_cast<int>(std::round(rough_length / S * recon_cfg().w1)));
    result = LNLib::NurbsCurve::LeastSquaresApproximation(recon_cfg().p_nurbs, points, ctrl_point_num, curve);
    assert(result);

    curve_obj.curve.nurbs = curve;
    curve_obj.curve.compute_arc_length();
    curve_obj.curve.arc_length /= SCALE;

    if(curve_obj.curve.arc_length < OBJ_MIN_LEN) return false;

    curve_obj.curve.pixels.clear();
    curve_obj.curve.pixels.reserve(curve_obj.region.size());
    for(const auto& p : curve_obj.region)
    {
        curve_obj.curve.pixels.emplace_back(cv::Point(p.x, p.y));
    }
    curve_obj.curve.points.clear();
    int cur_pt_num = static_cast<int>(5 * std::round(rough_length / scaled_image.rows * 100));
    for (int j = 0; j < cur_pt_num; ++j) {
        double s = static_cast<double>(j) / (cur_pt_num - 1);
        LNLib::XYZ p = LNLib::NurbsCurve::GetPointOnCurve(curve_obj.curve.nurbs, s);
        curve_obj.curve.points.emplace_back(cv::Point2d(p.X(), p.Y()));
    }
    return true;
}

void CurveDetector::ll_angle(const double& threshold, const unsigned int& n_bins)
#if !NOT_TEED
{
    img_width = scaled_image.cols;
    img_height = scaled_image.rows;

#if VISUALIZATION
    Mat gradient_image = Mat::zeros(scaled_image.size(), CV_8UC1);
    Mat angle_image = Mat::zeros(scaled_image.size(), CV_8UC1);
#endif
    Mat teed_edge_vis;

    cv::Mat color_img;
    cv::cvtColor(scaled_image, color_img, cv::COLOR_GRAY2BGR);

    pybind11::array_t<uint8_t> np_img(
        {color_img.rows, color_img.cols, color_img.channels()},
        {static_cast<ssize_t>(color_img.step[0]),
        static_cast<ssize_t>(color_img.step[1]),
        static_cast<ssize_t>(1)},
        color_img.data);

    pybind11::object np_out = edge_det_module_.attr("inference")(np_img);
    pybind11::buffer_info buf = np_out.cast<pybind11::array_t<uint8_t>>().request();
    cv::Mat output = cv::Mat(buf.shape[0], buf.shape[1], CV_8UC1, (uchar*)buf.ptr).clone();
    assert(output.cols == img_width && output.rows == img_height);
    teed_edge_vis = output.clone();

    cv::Mat inverted;
    cv::bitwise_not(output, inverted);
    modgrad = inverted;
    angles = Mat_<double>(scaled_image.size());
    angles.row(img_height - 1).setTo(NOTDEF);
    angles.col(img_width - 1).setTo(NOTDEF);

    double max_grad = -1;
    for(int y = 0; y < img_height - 1; ++y)
    {
        const uchar* scaled_image_row = scaled_image.ptr<uchar>(y);
        const uchar* next_scaled_image_row = scaled_image.ptr<uchar>(y+1);

        double* angles_row = angles.ptr<double>(y);
        double* modgrad_row = modgrad.ptr<double>(y);
        for(int x = 0; x < img_width-1; ++x)
        {
            int DA = next_scaled_image_row[x + 1] - scaled_image_row[x];
            int BC = scaled_image_row[x + 1] - next_scaled_image_row[x];
            int gx = DA + BC;
            int gy = DA - BC;
            double norm = modgrad_row[x];
            if (norm < 0.3 * 255)
            {
                angles_row[x] = NOTDEF;
                modgrad_row[x] = 0.0;
            }
            else
            {
                angles_row[x] = fastAtan2(float(gx), float(-gy)) * DEG_TO_RADS;
                if(angles_row[x] >= M_2__PI) std::cout<<"Abnormal!  angles_row[x] = "<<angles_row[x]<<std::endl;
                if (norm > max_grad) { max_grad = norm; }
            }
        }
    }

#if VISUALIZATION
    for(int y = 0; y < img_height - 1; ++y)
    {
        double* angles_row = angles.ptr<double>(y);
        double* modgrad_row = modgrad.ptr<double>(y);
        for(int x = 0; x < img_width-1; ++x)
        {
            if(modgrad_row[x] > 0.3 * 255)
            {

                gradient_image.at<uchar>(y, x) = modgrad_row[x];

                int angle_value = int((angles_row[x] / (2 * M_PI)) * 255);
                angle_image.at<uchar>(y, x) = std::min(angle_value, 255);
            }
        }
    }
    gradient_image_ = gradient_image.clone();
    angle_image_ = angle_image.clone();

    auto show_scaled = [](const std::string& name, const cv::Mat& img) {
        if (img.empty() || img.rows <= 0) return;
        double s = SHOW_SIZE / img.rows;
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(), s, s, cv::INTER_NEAREST);
        cv::imshow(name, resized);
    };
    if (!teed_edge_vis.empty())
    {
        cv::Mat teed_inv;
        cv::bitwise_not(teed_edge_vis, teed_inv);
        show_scaled("[TEED]", teed_inv);
    }
    {
        cv::Mat modgrad_u8;
        modgrad.convertTo(modgrad_u8, CV_8U);
        show_scaled("[modgrad]", modgrad_u8);
    }
    show_scaled("[ll_angle]", angle_image);
#if VISUALIZATION == 1
    cv::waitKey(1);
#elif VISUALIZATION == 2
    cv::waitKey(0);
#endif
#endif

    ordered_points.clear();
    double bin_coef = (max_grad > 0) ? double(n_bins - 1) / max_grad : 0;
    for(int y = 0; y < img_height - 1; ++y)
    {
        const double* modgrad_row = modgrad.ptr<double>(y);
        for(int x = 0; x < img_width - 1; ++x)
        {
            normPoint _point;
            int i = int(modgrad_row[x] * bin_coef);
            _point.p = Point(x, y);
            _point.norm = i;
            ordered_points.push_back(_point);
        }
    }

    std::stable_sort(ordered_points.begin(), ordered_points.end(), compare_norm);

}
#else
{
    img_width = scaled_image.cols;
    img_height = scaled_image.rows;

#if VISUALIZATION
    Mat gradient_image = Mat::zeros(scaled_image.size(), CV_8UC1);
    Mat angle_image = Mat::zeros(scaled_image.size(), CV_8UC1);
#endif

    cv::Mat_<double> angles = Mat_<double>(scaled_image.size());
    cv::Mat_<double> modgrad = Mat_<double>(scaled_image.size());
    angles.row(img_height - 1).setTo(NOTDEF);
    angles.col(img_width - 1).setTo(NOTDEF);

    double max_grad = -1;
    for(int y = 0; y < img_height - 1; ++y)
    {
        const uchar* scaled_image_row = scaled_image.ptr<uchar>(y);
        const uchar* next_scaled_image_row = scaled_image.ptr<uchar>(y+1);
        double* angles_row = angles.ptr<double>(y);
        double* modgrad_row = modgrad.ptr<double>(y);
        for(int x = 0; x < img_width-1; ++x)
        {
            int DA = next_scaled_image_row[x + 1] - scaled_image_row[x];
            int BC = scaled_image_row[x + 1] - next_scaled_image_row[x];
            int gx = DA + BC;
            int gy = DA - BC;
            double norm = std::sqrt((gx * gx + gy * gy) / 4.0);

            modgrad_row[x] = norm;

            if (norm <= threshold)
            {
                angles_row[x] = NOTDEF;
            }
            else
            {
                angles_row[x] = fastAtan2(float(gx), float(-gy)) * DEG_TO_RADS;
                if (norm > max_grad) { max_grad = norm; }
            }
        }
    }

#if VISUALIZATION
    for(int y = 0; y < img_height - 1; ++y)
    {
        double* angles_row = angles.ptr<double>(y);
        double* modgrad_row = modgrad.ptr<double>(y);
        for(int x = 0; x < img_width-1; ++x)
        {
            if(modgrad_row[x] > threshold)
            {

                gradient_image.at<uchar>(y, x) = std::min(int(modgrad_row[x] * 255.0 / max_grad), 255);

                int angle_value = int(((angles_row[x] + M_PI) / (2 * M_PI)) * 255);
                angle_image.at<uchar>(y, x) = std::min(angle_value, 255);
            }
        }
    }
    gradient_image_ = gradient_image.clone();
    angle_image_ = angle_image.clone();
#endif

    ordered_points.clear();
    double bin_coef = (max_grad > 0) ? double(n_bins - 1) / max_grad : 0;
    for(int y = 0; y < img_height - 1; ++y)
    {
        const double* modgrad_row = modgrad.ptr<double>(y);
        for(int x = 0; x < img_width - 1; ++x)
        {
            normPoint _point;
            int i = int(modgrad_row[x] * bin_coef);
            _point.p = Point(x, y);
            _point.norm = i;
            ordered_points.push_back(_point);
        }
    }

    std::stable_sort(ordered_points.begin(), ordered_points.end(), compare_norm);

}
#endif

void CurveDetector::region_grow(const Point2i& s, std::deque<RegionPoint>& reg,
                                      double& reg_angle, const double& prec)
{
    reg.clear();

    RegionPoint seed;
    seed.x = s.x;
    seed.y = s.y;
    seed.used = &used.at<uchar>(s);
    reg_angle = angles.at<double>(s);
    seed.angle = reg_angle;
    seed.modgrad = modgrad.at<double>(s);
    reg.push_back(seed);

    float sumdx = float(std::cos(reg_angle));
    float sumdy = float(std::sin(reg_angle));
    *seed.used = USED;

    for (size_t i = 0;i<reg.size();i++)
    {
        const RegionPoint& rpoint = reg[i];
        int xx_min = std::max(rpoint.x - 1, 0), xx_max = std::min(rpoint.x + 1, img_width - 1);
        int yy_min = std::max(rpoint.y - 1, 0), yy_max = std::min(rpoint.y + 1, img_height - 1);
        for(int yy = yy_min; yy <= yy_max; ++yy)
        {
            uchar* used_row = used.ptr<uchar>(yy);
            const double* angles_row = angles.ptr<double>(yy);
            const double* modgrad_row = modgrad.ptr<double>(yy);
            for(int xx = xx_min; xx <= xx_max; ++xx)
            {
                uchar& is_used = used_row[xx];
                if(is_used != USED &&
                   (isAligned(xx, yy, reg_angle, prec)))
                {
                    const double& angle = angles_row[xx];

                    is_used = USED;
                    RegionPoint region_point;
                    region_point.x = xx;
                    region_point.y = yy;
                    region_point.used = &is_used;
                    region_point.modgrad = modgrad_row[xx];
                    region_point.angle = angle;

                    reg.push_back(region_point);

                    sumdx += cos(float(angle));
                    sumdy += sin(float(angle));

                    reg_angle = fastAtan2(sumdy, sumdx) * DEG_TO_RADS;
                }
            }
        }
    }
}

void CurveDetector::region2rect(const std::deque<RegionPoint>& reg,
                                      const double reg_angle, const double prec, const double p, rect& rec) const
{

    double x = 0, y = 0, sum = 0;
    for(size_t i = 0; i < reg.size(); ++i)
    {
        const RegionPoint& pnt = reg[i];
        const double& weight = pnt.modgrad;

        x += double(pnt.x) * weight;
        y += double(pnt.y) * weight;
        sum += weight;
    }

    assert(sum > 0);

    x /= sum;
    y /= sum;

    double theta = get_theta(reg, x, y, reg_angle, prec);

    double dx = cos(theta);
    double dy = sin(theta);
    double l_min = 0, l_max = 0, w_min = 0, w_max = 0;

    for(size_t i = 0; i < reg.size(); ++i)
    {
        double regdx = double(reg[i].x) - x;
        double regdy = double(reg[i].y) - y;

        double l = regdx * dx + regdy * dy;
        double w = -regdx * dy + regdy * dx;

        if(l > l_max) l_max = l;
        else if(l < l_min) l_min = l;
        if(w > w_max) w_max = w;
        else if(w < w_min) w_min = w;
    }

    rec.x1 = x + l_min * dx;
    rec.y1 = y + l_min * dy;
    rec.x2 = x + l_max * dx;
    rec.y2 = y + l_max * dy;
    rec.width = w_max - w_min;
    rec.x = x;
    rec.y = y;
    rec.theta = theta;
    rec.dx = dx;
    rec.dy = dy;
    rec.prec = prec;
    rec.p = p;

    if(rec.width < 1.0) rec.width = 1.0;
}

double CurveDetector::get_theta(const std::deque<RegionPoint>& reg, const double& x,
                                      const double& y, const double& reg_angle, const double& prec) const
{
    double Ixx = 0.0;
    double Iyy = 0.0;
    double Ixy = 0.0;

    for(size_t i = 0; i < reg.size(); ++i)
    {
        const double& regx = reg[i].x;
        const double& regy = reg[i].y;
        const double& weight = reg[i].modgrad;
        double dx = regx - x;
        double dy = regy - y;
        Ixx += dy * dy * weight;
        Iyy += dx * dx * weight;
        Ixy -= dx * dy * weight;
    }

    assert(!(double_equal(Ixx, 0) && double_equal(Iyy, 0) && double_equal(Ixy, 0)));

    double lambda = 0.5 * (Ixx + Iyy - sqrt((Ixx - Iyy) * (Ixx - Iyy) + 4.0 * Ixy * Ixy));

    double theta = (fabs(Ixx)>fabs(Iyy))?
                    double(fastAtan2(float(lambda - Ixx), float(Ixy))):
                    double(fastAtan2(float(Ixy), float(lambda - Iyy)));
    theta *= DEG_TO_RADS;

    if(angle_diff(theta, reg_angle) > prec) { theta += CV_PI; }

    return theta;
}

bool CurveDetector::refine(std::deque<RegionPoint>& reg, double reg_angle, const Point2i& s,
                                 const double prec, double p, rect& rec, const double& density_th)
{
    double density = double(reg.size()) / (dist(rec.x1, rec.y1, rec.x2, rec.y2) * rec.width);

    if (density >= density_th) { return true; }

    double xc = double(reg[0].x);
    double yc = double(reg[0].y);
    const double& ang_c = reg[0].angle;
    double sum = 0, s_sum = 0;
    int n = 0;

    for (size_t i = 0; i < reg.size(); ++i)
    {
        *(reg[i].used) = NOTUSED;
        if (dist(xc, yc, reg[i].x, reg[i].y) < rec.width)
        {
            const double& angle = reg[i].angle;
            double ang_d = angle_diff_signed(angle, ang_c);
            sum += ang_d;
            s_sum += ang_d * ang_d;
            ++n;
        }
    }
    assert(n > 0);
    double mean_angle = sum / double(n);

    double tau = 2.0 * sqrt((s_sum - 2.0 * mean_angle * sum) / double(n) + mean_angle * mean_angle);

    region_grow(Point(s.x, s.y), reg, reg_angle, tau);

    if (reg.size() < 2) { return false; }

    region2rect(reg, reg_angle, prec, p, rec);
    density = double(reg.size()) / (dist(rec.x1, rec.y1, rec.x2, rec.y2) * rec.width);

    if (density < density_th)
    {
        return reduce_region_radius(reg, reg_angle, prec, p, rec, density, density_th);
    }
    else
    {
        return true;
    }
}

bool CurveDetector::reduce_region_radius(std::deque<RegionPoint>& reg, double reg_angle,
                const double prec, double p, rect& rec, double density, const double& density_th)
{

    double xc = double(reg[0].x);
    double yc = double(reg[0].y);
    double radSq1 = distSq(xc, yc, rec.x1, rec.y1);
    double radSq2 = distSq(xc, yc, rec.x2, rec.y2);
    double radSq = radSq1 > radSq2 ? radSq1 : radSq2;

    while(density < density_th)
    {
        radSq *= 0.75*0.75;

        for (size_t i = 0; i < reg.size(); ++i)
        {
            if(distSq(xc, yc, double(reg[i].x), double(reg[i].y)) > radSq)
            {

                *(reg[i].used) = NOTUSED;
                std::swap(reg[i], reg[reg.size() - 1]);
                reg.pop_back();
                --i;
            }
        }

        if(reg.size() < 2) { return false; }

        region2rect(reg ,reg_angle, prec, p, rec);

        density = double(reg.size()) /
                  (dist(rec.x1, rec.y1, rec.x2, rec.y2) * rec.width);
    }

    return true;
}

double CurveDetector::rect_improve(rect& rec) const
{
    double delta = 0.5;
    double delta_2 = delta / 2.0;

    double log_nfa = rect_nfa(rec);

    if(log_nfa > LOG_EPS) return log_nfa;

    rect r = rect(rec);
    for(int n = 0; n < 5; ++n)
    {
        r.p /= 2;
        r.prec = r.p * CV_PI;
        double log_nfa_new = rect_nfa(r);
        if(log_nfa_new > log_nfa)
        {
            log_nfa = log_nfa_new;
            rec = rect(r);
        }
    }
    if(log_nfa > LOG_EPS) return log_nfa;

    r = rect(rec);
    for(unsigned int n = 0; n < 5; ++n)
    {
        if((r.width - delta) >= 0.5)
        {
            r.width -= delta;
            double log_nfa_new = rect_nfa(r);
            if(log_nfa_new > log_nfa)
            {
                rec = rect(r);
                log_nfa = log_nfa_new;
            }
        }
    }
    if(log_nfa > LOG_EPS) return log_nfa;

    r = rect(rec);
    for(unsigned int n = 0; n < 5; ++n)
    {
        if((r.width - delta) >= 0.5)
        {
            r.x1 += -r.dy * delta_2;
            r.y1 +=  r.dx * delta_2;
            r.x2 += -r.dy * delta_2;
            r.y2 +=  r.dx * delta_2;
            r.width -= delta;
            double log_nfa_new = rect_nfa(r);
            if(log_nfa_new > log_nfa)
            {
                rec = rect(r);
                log_nfa = log_nfa_new;
            }
        }
    }
    if(log_nfa > LOG_EPS) return log_nfa;

    r = rect(rec);
    for(unsigned int n = 0; n < 5; ++n)
    {
        if((r.width - delta) >= 0.5)
        {
            r.x1 -= -r.dy * delta_2;
            r.y1 -=  r.dx * delta_2;
            r.x2 -= -r.dy * delta_2;
            r.y2 -=  r.dx * delta_2;
            r.width -= delta;
            double log_nfa_new = rect_nfa(r);
            if(log_nfa_new > log_nfa)
            {
                rec = rect(r);
                log_nfa = log_nfa_new;
            }
        }
    }
    if(log_nfa > LOG_EPS) return log_nfa;

    r = rect(rec);
    for(unsigned int n = 0; n < 5; ++n)
    {
        if((r.width - delta) >= 0.5)
        {
            r.p /= 2;
            r.prec = r.p * CV_PI;
            double log_nfa_new = rect_nfa(r);
            if(log_nfa_new > log_nfa)
            {
                rec = rect(r);
                log_nfa = log_nfa_new;
            }
        }
    }

    return log_nfa;
}

double CurveDetector::rect_nfa(const rect& rec) const
{
    int total_pts = 0, alg_pts = 0;
    double half_width = rec.width / 2.0;
    double dyhw = rec.dy * half_width;
    double dxhw = rec.dx * half_width;

    cv::Point2d v_tmp[4];
    v_tmp[0] = cv::Point2d(rec.x1 - dyhw, rec.y1 + dxhw);
    v_tmp[1] = cv::Point2d(rec.x2 - dyhw, rec.y2 + dxhw);
    v_tmp[2] = cv::Point2d(rec.x2 + dyhw, rec.y2 - dxhw);
    v_tmp[3] = cv::Point2d(rec.x1 + dyhw, rec.y1 - dxhw);

    int offset = 0;
    for (int i = 1; i < 4; ++i) {
        if (AsmallerB_YoverX(v_tmp[i], v_tmp[offset])){
            offset = i;
        }
    }

    cv::Point2d ordered_y[4];
    for (int i = 0; i < 4; ++i) {
        ordered_y[i] = v_tmp[(i + offset) % 4];
    }

    double flstep = get_slope(ordered_y[0], ordered_y[1]);
    double slstep = get_slope(ordered_y[1], ordered_y[2]);

    double frstep = get_slope(ordered_y[0], ordered_y[3]);
    double srstep = get_slope(ordered_y[3], ordered_y[2]);

    double top_y = ordered_y[0].y, bottom_y = ordered_y[2].y;

    std::vector<cv::Point> points;
    double left_limit, right_limit;
    for(int y = (int) ceil(top_y); y <= (int) ceil(bottom_y); ++y)
    {
        if (y < 0 || y >= img_height) continue;

        if(y <= int(ceil(ordered_y[1].y)))
            left_limit = get_limit(ordered_y[0], y, flstep);
        else
            left_limit = get_limit(ordered_y[1], y, slstep);

        if(y < int(ceil(ordered_y[3].y)))
            right_limit = get_limit(ordered_y[0], y, frstep);
        else
            right_limit = get_limit(ordered_y[3], y, srstep);

        for(int x = (int) ceil(left_limit); x <= (int)(right_limit); ++x) {
            if (x < 0 || x >= img_width) continue;

            ++total_pts;
            if(isAligned(x, y, rec.theta, rec.prec))
            {
                ++alg_pts;
            }
        }
    }

    return nfa(total_pts, alg_pts, rec.p);
}

double CurveDetector::nfa(const int& n, const int& k, const double& p) const
{

    if(n == 0 || k == 0) { return -LOG_NT; }
    if(n == k) { return -LOG_NT - double(n) * log10(p); }

    double p_term = p / (1 - p);

    double log1term = log_gamma(double(n) + 1) - log_gamma(double(k) + 1)
                - log_gamma(double(n-k) + 1)
                + double(k) * log(p) + double(n-k) * log(1.0 - p);
    double term = exp(log1term);

    if(double_equal(term, 0))
    {
        if(k > n * p) return -log1term / M_LN10 - LOG_NT;
        else return -LOG_NT;
    }

    double bin_tail = term;
    double tolerance = 0.1;
    for(int i = k + 1; i <= n; ++i)
    {
        double bin_term = double(n - i + 1) / double(i);
        double mult_term = bin_term * p_term;
        term *= mult_term;
        bin_tail += term;
        if(bin_term < 1)
        {
            double err = term * ((1 - pow(mult_term, double(n-i+1))) / (1 - mult_term) - 1);
            if(err < tolerance * fabs(-log10(bin_tail) - LOG_NT) * bin_tail) break;
        }

    }
    return -log10(bin_tail) - LOG_NT;
}

inline bool CurveDetector::isAligned(int x, int y, const double& theta, const double& prec) const
{
    if(x < 0 || y < 0 || x >= angles.cols || y >= angles.rows) { return false; }
    const double& a = angles.at<double>(y, x);
    if(a == NOTDEF) { return false; }

    double delta = angle_diff(theta, a);
#if SHOW_REGION_GROW
    std::cout<<"Window theta = "<<theta / DEG_TO_RADS<<" deg"<<std::endl;
    std::cout<<"Measured angle = "<<a / DEG_TO_RADS<<" deg"<<std::endl;
    std::cout<<"Angle difference = "<<delta / DEG_TO_RADS<<" deg"<<std::endl;
    std::cout<<"         prec = "<<prec / DEG_TO_RADS<<" deg"<<std::endl;
    std::cout<<"Angles aligned?  "<<(delta <= prec? "Yes" : "No")<<std::endl<<std::endl;
#endif

    return delta <= prec;
}

void CurveDetector::drawObjects_nurbs(const std::vector<Object>& objs, const std::string& img_name)
{

    cv::Mat mutable_image;
    if (image.channels() == 1) {
        cv::cvtColor(image, mutable_image, cv::COLOR_GRAY2BGR);
    } else {
        mutable_image = image.clone();
    }

    cv::RNG rng(cv::getTickCount());

    for(const auto& obj : objs)
    {
        if(obj.is_curve)
        {
            cv::Scalar color = cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
            LNLib::XYZ pt_last = LNLib::NurbsCurve::GetPointOnCurve(obj.curve.nurbs, 0.0);
            double u_last = pt_last.X(),
                   v_last = pt_last.Y();
            for(double s = 0.01; s < 1.0; s += 0.01)
            {
                LNLib::XYZ pt = LNLib::NurbsCurve::GetPointOnCurve(obj.curve.nurbs, s);
                double u = pt.X(),
                       v = pt.Y();
                cv::line(mutable_image, cv::Point2f(u_last, v_last), cv::Point2f(u, v), color, LINE_WIDTH);
                u_last = u; v_last = v;
            }
        }
        else
        {

            cv::Scalar color = cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
            cv::line(mutable_image, cv::Point2f(obj.rec.x1, obj.rec.y1), cv::Point2f(obj.rec.x2, obj.rec.y2), color, LINE_WIDTH);
        }
    }

    double _scale = SHOW_SIZE / mutable_image.rows;
    cv::Mat image_resized;
    cv::resize(mutable_image, image_resized, cv::Size(), _scale, _scale, cv::INTER_NEAREST);
    cv::imshow(img_name, image_resized);

#if VISUALIZATION == 1
    cv::waitKey(1);
#elif VISUALIZATION == 2
    cv::waitKey(0);
#endif

}
