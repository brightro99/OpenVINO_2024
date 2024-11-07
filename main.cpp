#include <fstream>
#include <iostream>
#include <string>

#include "opencv2/opencv.hpp"
#include "openvino/core/preprocess/input_info.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/openvino.hpp"

struct BaseObject {
    cv::Rect bbox;
    float confidence;
};

struct ImageResizeParams {
    float scale_factor;
    cv::Size2f pad;
};

template <typename T>
static void NMS(std::vector<T>& objects, const float iou_threshold, bool use_bbox_size_filter = false) {
    if (objects.empty())
        return;

    /* 1차 정제: Confidence 순으로 정렬 */
    std::sort(objects.begin(), objects.end(), [](const T& lhs, const T& rhs) { return lhs.confidence > rhs.confidence; });

    size_t valid_count = 0;

    for (size_t i = 0; i < objects.size(); ++i) {
        bool keep = true;
        for (size_t j = 0; j < valid_count; ++j) {
            float iou = calculateIoU(objects[j].bbox, objects[i].bbox);
            if (iou > iou_threshold) {
                keep = false;
                break;
            }
        }

        if (keep) {
            if (valid_count != i) {
                objects[valid_count] = std::move(objects[i]);
            }
            ++valid_count;
        }
    }

    objects.resize(valid_count);

    if (use_bbox_size_filter) {
        /* 2차 정제: Box 크기로 정제 */
        valid_count = 0;
        std::sort(objects.begin(), objects.end(), [](T& lhs, T& rhs) { return lhs.bbox.area() > rhs.bbox.area(); });

        for (size_t i = 0; i < objects.size(); ++i) {
            bool keep = true;
            for (size_t j = 0; j < valid_count; ++j) {
                float iou = calculateIoU(objects[j].bbox, objects[i].bbox);
                if (iou > iou_threshold) {
                    keep = false;
                    break;
                }
            }

            if (keep) {
                if (valid_count != i) {
                    objects[valid_count] = std::move(objects[i]);
                }
                ++valid_count;
            }
        }
        objects.resize(valid_count);
    }
}

template <typename T>
static float calculateIoU(const T& obj1, const T& obj2) {
    int obj1_area = obj1.width * obj1.height;
    int obj2_area = obj2.width * obj2.height;
    int inter_x0 = std::max(static_cast<int>(obj1.x), static_cast<int>(obj2.x));
    int inter_y0 = std::max(static_cast<int>(obj1.y), static_cast<int>(obj2.y));
    int inter_x1 = std::min(static_cast<int>(obj1.x + obj1.width), static_cast<int>(obj2.x + obj2.width));
    int inter_y1 = std::min(static_cast<int>(obj1.y + obj1.height), static_cast<int>(obj2.y + obj2.height));
    if (inter_x1 < inter_x0 || inter_y1 < inter_y0)
        return 0.f;

    if (isOverlapped(obj1, obj2))
        return 1.f;

    int area_inter = (inter_x1 - inter_x0) * (inter_y1 - inter_y0);
    int area_sum = obj1_area + obj2_area - area_inter;

    return static_cast<float>(area_inter) / area_sum;
}

template <typename T>
static bool isOverlapped(const T& obj1, const T& obj2) {
    return (
        obj1.x <= obj2.x &&
        obj1.y <= obj2.y &&
        (obj1.x + obj1.width) >= (obj2.x + obj2.width) &&
        (obj1.y + obj1.height) >= (obj2.y + obj2.height));
}

void printInputAndOutputsInfo(const ov::Model& network) {
    std::cout << "model name: " << network.get_friendly_name() << std::endl;

    const std::vector<ov::Output<const ov::Node>> inputs = network.inputs();
    for (const ov::Output<const ov::Node>& input : inputs) {
        std::cout << "    inputs" << std::endl;

        const std::string name = input.get_names().empty() ? "NONE" : input.get_any_name();
        std::cout << "        input name: " << name << std::endl;

        const ov::element::Type type = input.get_element_type();
        std::cout << "        input type: " << type << std::endl;

        const ov::Shape shape = input.get_shape();
        std::cout << "        input shape: " << shape << std::endl;
    }

    const std::vector<ov::Output<const ov::Node>> outputs = network.outputs();
    for (const ov::Output<const ov::Node>& output : outputs) {
        std::cout << "    outputs" << std::endl;

        const std::string name = output.get_names().empty() ? "NONE" : output.get_any_name();
        std::cout << "        output name: " << name << std::endl;

        const ov::element::Type type = output.get_element_type();
        std::cout << "        output type: " << type << std::endl;

        const ov::Shape shape = output.get_shape();
        std::cout << "        output shape: " << shape << std::endl;
    }
}

std::string readFileToString(const std::string& file_path) {
    std::ifstream file(file_path);

    if (!file) {
        throw std::runtime_error("파일을 열 수 없습니다: " + file_path);
    }

    std::ostringstream buffer;
    buffer << file.rdbuf();  // 파일의 전체 내용을 스트림으로 읽어오기
    return buffer.str();     // std::string으로 반환
}

std::vector<uint8_t> readFileToBinary(const std::string& file_path) {
    // 바이너리 모드로 파일을 엽니다.
    std::ifstream file(file_path, std::ios::binary | std::ios::ate);

    if (!file) {
        throw std::runtime_error("파일을 열 수 없습니다: " + file_path);
    }

    // 파일 크기를 구하고, 해당 크기만큼의 버퍼를 할당합니다.
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(size);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        throw std::runtime_error("파일을 읽는 데 실패했습니다: " + file_path);
    }

    return buffer;
}

void generateScale(cv::Mat& image, const cv::Size2f& target_size, float& scale_factor) {
    int origin_w = image.cols;
    int origin_h = image.rows;

    int target_h = target_size.height;
    int target_w = target_size.width;

    scale_factor = std::min(static_cast<float>(target_h) / static_cast<float>(origin_h),
                            static_cast<float>(target_w) / static_cast<float>(origin_w));
}

void letterbox(cv::Mat& input_image, cv::Mat& output_image, const cv::Size2f& target_size, ImageResizeParams& prarms) {
    if (input_image.cols == target_size.width && input_image.rows == target_size.height) {
        if (input_image.data == output_image.data) {
            prarms.scale_factor = 1.0;
            prarms.pad.width = 0;
            prarms.pad.height = 0;
            return;
        } else {
            prarms.scale_factor = 1.0;
            prarms.pad.width = 0;
            prarms.pad.height = 0;
            output_image = input_image.clone();
            return;
        }
    }

    generateScale(input_image, target_size, prarms.scale_factor);
    int new_shape_w = std::round(input_image.cols * prarms.scale_factor);
    int new_shape_h = std::round(input_image.rows * prarms.scale_factor);
    prarms.pad.width = (target_size.width - new_shape_w) / 2.;
    prarms.pad.height = (target_size.height - new_shape_h) / 2.;

    int top = std::round(prarms.pad.height - 0.1);
    int bottom = std::round(prarms.pad.height + 0.1);
    int left = std::round(prarms.pad.width - 0.1);
    int right = std::round(prarms.pad.width + 0.1);

    cv::resize(input_image, output_image,
               cv::Size(new_shape_w, new_shape_h),
               0, 0, cv::INTER_AREA);

    cv::copyMakeBorder(output_image, output_image, top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar(114.));
}

int main(int argc, char const* argv[]) {
    std::cout << "Hello World" << std::endl;
    cv::Mat frame = cv::imread("C:\\Users\\brightro99\\Desktop\\Workspace\\OpenVINO_2024\\images\\sonny.jpg");

    std::string weights_path = "C:\\Users\\brightro99\\Desktop\\Workspace\\OpenVINO_2024\\models\\pe-01.bin";
    std::string xml_path = "C:\\Users\\brightro99\\Desktop\\Workspace\\OpenVINO_2024\\models\\pe-01.xml";

    ov::Core core;

    std::cout << "Device info: " << std::endl;
    std::string device_name = "CPU";
    bool use_npu = false;
    if (use_npu) {
        device_name = "NPU";
    }

    std::cout << core.get_versions(device_name) << std::endl;

    try {
        std::string xml = readFileToString(xml_path);
        std::vector<uint8_t> weights = readFileToBinary(weights_path);

        ov::Tensor weights_tensor = ov::Tensor(ov::element::u8, {1, weights.size()}, weights.data());
        std::shared_ptr<ov::Model> model = core.read_model(xml, weights_tensor);
        printInputAndOutputsInfo(*model);

        // Preprocessing setup for the model
        /*
            <layer id="0" name="images" type="Parameter" version="opset1">
			    <data shape="1,3,640,640" element_type="f32" />
			    <output>
				    <port id="0" precision="FP32" names="images">
					    <dim>1</dim>
					    <dim>3</dim>
					    <dim>640</dim>
					    <dim>640</dim>
				    </port>
			    </output>
		    </layer>        
        */
        ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);
        /*
        * 입력 이미지의 기본 속성 정의
        * 8비트 정수형, NHWC, BGR 이미지
        */
        ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::BGR);

        /*
        * 모델 입력 텐서의 기본 속성 정의
        * 32비트 부동소수점, RGB, Scale 255 (0 ~ 1 사이로 Norm)
        */
        ppp.input().preprocess().convert_element_type(ov::element::f32).convert_color(ov::preprocess::ColorFormat::RGB).scale({255, 255, 255});

        /*
        * 모델 입력 레이어 설정
        * NCHW
        */
        ppp.input().model().set_layout("NCHW");

        /*
        * 모델 출력 타입 설정
        * 32비트 부동소수점
        */
        ppp.output().tensor().set_element_type(ov::element::f32);

        model = ppp.build();  // Build the preprocessed model;

        ov::InferRequest inference_request_;  // OpenVINO inference request
        ov::CompiledModel compiled_model_;    // OpenVINO compiled model

        // Compile the model for inference
        compiled_model_ = core.compile_model(model, device_name);
        inference_request_ = compiled_model_.create_infer_request();  // Create inference request

        ImageResizeParams resize_params_;
        cv::Size2f model_input_shape_;  // Input shape of the model
        cv::Size model_output_shape_;   // Output shape of the model

        float model_confidence_threshold_ = 0.5;  // Confidence threshold for detections
        float model_NMS_threshold_ = 0.4;         // Non-Maximum Suppression threshold

        // Get input shape from the model
        const std::vector<ov::Output<ov::Node>> inputs = model->inputs();
        const ov::Shape input_shape = inputs[0].get_shape();
        model_input_shape_ = cv::Size2f(input_shape[2], input_shape[1]);  // W, H

        // Get output shape from the model
        const std::vector<ov::Output<ov::Node>> outputs = model->outputs();
        const ov::Shape output_shape = outputs[0].get_shape();
        model_output_shape_ = cv::Size(output_shape[2], output_shape[1]);  // W, H

        cv::Mat resized_frame;
        letterbox(frame, resized_frame, model_input_shape_, resize_params_);

        //cv::imshow("resize.jpg", resized_frame);
        //cv::waitKey(0);

        float* input_data = (float*)resized_frame.data;                                                                                           // Get pointer to resized frame data
        const ov::Tensor input_tensor = ov::Tensor(compiled_model_.input().get_element_type(), compiled_model_.input().get_shape(), input_data);  // Create input tensor
        inference_request_.set_input_tensor(input_tensor);                                                                                        // Set input tensor for inference

        inference_request_.infer();  // Run inference

        std::vector<BaseObject> objects;
        // Get the output tensor from the inference request
        const float* detections = inference_request_.get_output_tensor().data<const float>();
        const cv::Mat detection_outputs(model_output_shape_, CV_32F, (float*)detections);  // Create OpenCV matrix from output tensor

        std::cout << detection_outputs.cols << " " << detection_outputs.rows << " " << detection_outputs.channels() << std::endl; 

        // 8400, 56, 1

        for (int i = 0; i < detection_outputs.cols; ++i) {

            float confidence = detection_outputs.at<float>(4, i);
            if (confidence < model_confidence_threshold_) {
                continue;
            }

            BaseObject object;
            object.confidence = confidence;

            cv::Rect box(
                static_cast<int>(detection_outputs.at<float>(0, i) - detection_outputs.at<float>(2, i) / 2),  // left
                static_cast<int>(detection_outputs.at<float>(1, i) - detection_outputs.at<float>(3, i) / 2),  // top
                static_cast<int>(detection_outputs.at<float>(2, i)),                                          // width
                static_cast<int>(detection_outputs.at<float>(3, i))                                           // height
            );

            
            float x1 = static_cast<float>(box.x);
            float y1 = static_cast<float>(box.y);
            float x2 = static_cast<float>(box.x + box.width);
            float y2 = static_cast<float>(box.y + box.height);

            x1 -= resize_params_.pad.width;
            y1 -= resize_params_.pad.height;
            x2 -= resize_params_.pad.width;
            y2 -= resize_params_.pad.height;

            x1 /= resize_params_.scale_factor;
            y1 /= resize_params_.scale_factor;
            x2 /= resize_params_.scale_factor;
            y2 /= resize_params_.scale_factor;

            box.x = (int)x1;
            box.y = (int)y1;
            box.width = (int)(x2 - x1);
            box.height = (int)(y2 - y1);

            object.bbox = box;

            //std::cout << object.confidence << std::endl;
            //std::cout << object.bbox.x << std::endl;
            //std::cout << object.bbox.y << std::endl;
            //std::cout << object.bbox.width << std::endl;
            //std::cout << object.bbox.height << std::endl;
            objects.push_back(object);
        }

        // NMS
        NMS(objects, model_NMS_threshold_);

        for (auto& item : objects) {
            cv::rectangle(frame, item.bbox, cv::Scalar(0, 0, 255), 2);
            std::string person_label = "Conf : " + std::to_string(item.confidence);

            int body_text_x = item.bbox.x + 5;
            int body_text_y = item.bbox.y + 18;
            if (body_text_y < 0) {
                body_text_y = item.bbox.y + item.bbox.height - 18;
            }
            if (body_text_x < 0) {
                body_text_x = 5;
            } else if (body_text_x + item.bbox.width > frame.cols) {
                body_text_x = frame.cols - item.bbox.width;
            }
            cv::putText(frame, person_label, cv::Point(body_text_x, body_text_y), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 1, 1);
        }
        //cv::imwrite("result.jpg", frame);
        cv::imshow("result.jpg", frame);
        cv::waitKey(0);

    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }

    return 0;
}
