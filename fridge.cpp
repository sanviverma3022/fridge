#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

// Define a struct to represent an item in the fridge
struct FridgeItem {
    std::string name;
    // Add other relevant attributes here like expiration date, quantity, etc.
};

// Function to perform object detection and scan items in the fridge
std::vector<FridgeItem> scanFridgeItems(const cv::Mat& fridgeImage) {
    std::vector<FridgeItem> items;
    
    // Load pre-trained object detection model (e.g., YOLO)
    cv::dnn::Net net = cv::dnn::readNet("yolov3.weights", "yolov3.cfg");
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // Get names of output layers
    std::vector<std::string> outputLayerNames = net.getUnconnectedOutLayersNames();

    // Prepare input blob
    cv::Mat blob;
    cv::dnn::blobFromImage(fridgeImage, blob, 1.0 / 255.0, cv::Size(416, 416), cv::Scalar(0, 0, 0), true, false);

    // Set input to the network
    net.setInput(blob);

    // Forward pass through the network
    std::vector<cv::Mat> outs;
    net.forward(outs, outputLayerNames);

    // Post-processing
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    for (const auto& out : outs) {
        // Loop over each detection
        for (int i = 0; i < out.rows; ++i) {
            cv::Mat scores = out.row(i).colRange(5, out.cols);
            cv::Point classIdPoint;
            double confidence;
            // Get the class and confidence
            cv::minMaxLoc(scores, nullptr, &confidence, nullptr, &classIdPoint);
            // Filter out weak predictions
            if (confidence > 0.5) {
                int centerX = static_cast<int>(out.at<float>(i, 0) * fridgeImage.cols);
                int centerY = static_cast<int>(out.at<float>(i, 1) * fridgeImage.rows);
                int width = static_cast<int>(out.at<float>(i, 2) * fridgeImage.cols);
                int height = static_cast<int>(out.at<float>(i, 3) * fridgeImage.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                classIds.push_back(classIdPoint.x);
                confidences.push_back(static_cast<float>(confidence));
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }

    // Non-maxima suppression
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, 0.5, 0.4, indices);

    // Extract detected items
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        FridgeItem item;
        item.name = "Unknown"; // Placeholder for the item name
        items.push_back(item);
    }

    return items;
}

// Function to suggest recipes based on the items in the fridge
void suggestRecipes(const std::vector<FridgeItem>& items) {
    // Dummy recipe database for demonstration
    std::map<std::vector<std::string>, std::string> recipes = {
        {{"Apple"}, "Apple Salad"},
        {{"Carrot"}, "Carrot Soup"},
        {{"Milk"}, "Cereal with Milk"}
        // Add more recipes here
    };

    // Find recipes that can be made with the items in the fridge
    std::vector<std::string> availableIngredients;
    for (const auto& item : items) {
        availableIngredients.push_back(item.name);
    }

    bool recipeFound = false;
    for (const auto& recipe : recipes) {
        const auto& ingredients = recipe.first;
        bool recipePossible = true;
        for (const auto& ingredient : ingredients) {
            auto it = std::find(availableIngredients.begin(), availableIngredients.end(), ingredient);
            if (it == availableIngredients.end()) {
                recipePossible = false;
                break;
            }
        }
        if (recipePossible) {
            std::cout << "You can make: " << recipe.second << std::endl;
            recipeFound = true;
        }
    }
    if (!recipeFound) {
        std::cout << "No recipes found with the available ingredients." << std::endl;
    }
}

int main() {
    std::cout << "Smart Fridge App\n" << std::endl;

    // Example: Load an image of the fridge
    cv::Mat fridgeImage = cv::imread("fridge_image.jpg");

    // Check if the image was loaded successfully
    if (fridgeImage.empty()) {
        std::cerr << "Error: Failed to load fridge image." << std::endl;
        return 1;
    }

    // Simulate scanning the fridge image and getting the list of items
    std::cout << "Scanning items in the fridge..." << std::endl;
    std::vector<FridgeItem> items = scanFridgeItems(fridgeImage);

    // Display the items found in the fridge
    std::cout << "\nItems in the fridge:" << std::endl;
    if (items.empty()) {
        std::cout << "No items found in the fridge." << std::endl;
    } else {
        for (const auto& item : items) {
            std::cout << "- " << item.name << std::endl;
        }
    }

    // Suggest recipes based on the items in the fridge
    std::cout << "\nSuggested recipes:" << std::endl;
    suggestRecipes(items);

    return 0;
}

