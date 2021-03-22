/*
 * YoloObjectDetector.cpp
 *
 *  Created on: Dec 19, 2016
 *      Author: Marko Bjelonic
 *   Institute: ETH Zurich, Robotic Systems Lab
 */

// yolo object detector
#include "darknet_ros/YoloObjectDetector.hpp"

// Check for xServer
#include <X11/Xlib.h>

#ifdef DARKNET_FILE_PATH
std::string darknetFilePath_ = DARKNET_FILE_PATH;
#else
#error Path of darknet repository is not defined in CMakeLists.txt.
#endif

namespace darknet_ros {

char *cfg;
char *weights;
char *data;
char **detectionNames;

YoloObjectDetector::YoloObjectDetector(ros::NodeHandle nh)
    : nodeHandle_(nh),
      numClasses_(0),
      classLabels_(0),
      imageTransport_(nodeHandle_),
      rosBoxes_(0),
      rosBoxCounter_(0)
{
  ROS_INFO("[YoloObjectDetector] Node started.");

  // Read parameters from config file.
  if (!readParameters()) {
    ros::requestShutdown();
  }

  init();
}

YoloObjectDetector::~YoloObjectDetector()
{
  {
    boost::unique_lock<boost::shared_mutex> lockNodeStatus(mutexNodeStatus_);
    isNodeRunning_ = false;
  }
  yoloThread_.join();
}

bool YoloObjectDetector::readParameters()
{
  // Load common parameters.
  nodeHandle_.param("image_view/enable_opencv", viewImage_, true);
  nodeHandle_.param("image_view/wait_key_delay", waitKeyDelay_, 3);
  nodeHandle_.param("image_view/enable_console_output", enableConsoleOutput_, false);

  // Check if Xserver is running on Linux.
  if (XOpenDisplay(NULL)) {
    // Do nothing!
    ROS_INFO("[YoloObjectDetector] Xserver is running.");
  } else {
    ROS_INFO("[YoloObjectDetector] Xserver is not running.");
    viewImage_ = false;
  }

  // Set vector sizes.
  nodeHandle_.param("yolo_model/detection_classes/names", classLabels_,
                    std::vector<std::string>(0));
  numClasses_ = classLabels_.size();
  rosBoxes_ = std::vector<std::vector<RosBox_> >(numClasses_);
  rosBoxCounter_ = std::vector<int>(numClasses_);

  return true;
}

void YoloObjectDetector::init()
{
  ROS_INFO("[YoloObjectDetector] init().");

  // Initialize deep network of darknet.
  std::string weightsPath;
  std::string configPath;
  std::string dataPath;
  std::string configModel;
  std::string weightsModel;

  // Threshold of object detection.
  float thresh;
  nodeHandle_.param("yolo_model/threshold/value", thresh, (float) 0.3);

  // Path to weights file.
  nodeHandle_.param("yolo_model/weight_file/name", weightsModel,
                    std::string("yolov2-tiny.weights"));
  nodeHandle_.param("weights_path", weightsPath, std::string("/default"));
  weightsPath += "/" + weightsModel;
  weights = new char[weightsPath.length() + 1];
  strcpy(weights, weightsPath.c_str());

  // Path to config file.
  nodeHandle_.param("yolo_model/config_file/name", configModel, std::string("yolov2-tiny.cfg"));
  nodeHandle_.param("config_path", configPath, std::string("/default"));
  configPath += "/" + configModel;
  cfg = new char[configPath.length() + 1];
  strcpy(cfg, configPath.c_str());

  // Path to data folder.
  dataPath = darknetFilePath_;
  dataPath += "/data";
  data = new char[dataPath.length() + 1];
  strcpy(data, dataPath.c_str());

  // Get classes.
  detectionNames = (char**) realloc((void*) detectionNames, (numClasses_ + 1) * sizeof(char*));
  for (int i = 0; i < numClasses_; i++) {
    detectionNames[i] = new char[classLabels_[i].length() + 1];
    strcpy(detectionNames[i], classLabels_[i].c_str());
  }

  // Load network.
  setupNetwork(cfg, weights, data, thresh, detectionNames, numClasses_,
                0, 0, 1, 0.5, 0, 0, 0, 0);

  nodeHandle_.param("publishers/rate_hz", publishRate_, 10);
  nodeHandle_.param("nr_scouts", nr_scouts, 1);
  nodeHandle_.param("nr_excavators", nr_excavators, 1);
  nodeHandle_.param("nr_haulers", nr_haulers, 1);


  imageSubscribers_ = std::vector<image_transport::Subscriber>(nr_scouts + nr_excavators + nr_haulers);
  boundingBoxesPublishers_ = std::vector<ros::Publisher>(nr_scouts + nr_excavators + nr_haulers);
  detectionImagePublishers_ = std::vector<ros::Publisher> (nr_scouts + nr_excavators + nr_haulers);
  objectPublishers_ = std::vector<ros::Publisher> (nr_scouts + nr_excavators + nr_haulers);

  imageHeaders_ = std::vector<std_msgs::Header>  (nr_scouts + nr_excavators + nr_haulers);
  camImageCopies_ = std::vector<cv::Mat> (nr_scouts + nr_excavators + nr_haulers);
  imageStatus_ = std::vector<bool> (nr_scouts + nr_excavators + nr_haulers, false);

  yoloThread_ = std::thread(&YoloObjectDetector::yolo, this);

  // Initialize publisher and subscriber.
  std::string cameraTopicName;
  int cameraQueueSize;
  std::string objectDetectorTopicName;
  int objectDetectorQueueSize;
  bool objectDetectorLatch;
  std::string boundingBoxesTopicName;
  int boundingBoxesQueueSize;
  bool boundingBoxesLatch;
  std::string detectionImageTopicName;
  int detectionImageQueueSize;
  bool detectionImageLatch;


  nodeHandle_.param("subscribers/camera_reading/topic", cameraTopicName,
                    std::string("/camera/image_raw"));
  nodeHandle_.param("subscribers/camera_reading/queue_size", cameraQueueSize, 1);
  nodeHandle_.param("publishers/object_detector/topic", objectDetectorTopicName,
                    std::string("found_object"));
  nodeHandle_.param("publishers/object_detector/queue_size", objectDetectorQueueSize, 1);
  nodeHandle_.param("publishers/object_detector/latch", objectDetectorLatch, false);
  nodeHandle_.param("publishers/bounding_boxes/topic", boundingBoxesTopicName,
                    std::string("bounding_boxes"));
  nodeHandle_.param("publishers/bounding_boxes/queue_size", boundingBoxesQueueSize, 1);
  nodeHandle_.param("publishers/bounding_boxes/latch", boundingBoxesLatch, false);
  nodeHandle_.param("publishers/detection_image/topic", detectionImageTopicName,
                    std::string("detection_image"));
  nodeHandle_.param("publishers/detection_image/queue_size", detectionImageQueueSize, 1);
  nodeHandle_.param("publishers/detection_image/latch", detectionImageLatch, true);


  // http://ros-users.122217.n3.nabble.com/NodeHandle-subscribe-callbacks-td942195.html
  int cnt = 0;
  for (int i = 0; i < nr_scouts; i++) {
    std::string robot_name = "small_scout_" + boost::lexical_cast<std::string>(i+1);
    imageSubscribers_[cnt] = imageTransport_.subscribe("/" + robot_name + cameraTopicName, cameraQueueSize, boost::bind(&YoloObjectDetector::cameraCallback, this, _1, cnt));
    boundingBoxesPublishers_[cnt] = nodeHandle_.advertise<darknet_ros_msgs::BoundingBoxes>("/" + robot_name + boundingBoxesTopicName, boundingBoxesQueueSize, boundingBoxesLatch);
    detectionImagePublishers_[cnt] = nodeHandle_.advertise<sensor_msgs::Image>("/" + robot_name + detectionImageTopicName, detectionImageQueueSize, detectionImageLatch);
    objectPublishers_[cnt] = nodeHandle_.advertise<darknet_ros_msgs::ObjectCount>("/" + robot_name + objectDetectorTopicName, objectDetectorQueueSize, objectDetectorLatch);
    cnt++;
  }
  for (int i = 0; i < nr_excavators; i++) {
    std::string robot_name = "small_excavator_" + boost::lexical_cast<std::string>(i+1);
    imageSubscribers_[cnt]= imageTransport_.subscribe("/" + robot_name + cameraTopicName, cameraQueueSize, boost::bind(&YoloObjectDetector::cameraCallback, this, _1, cnt));
    boundingBoxesPublishers_[cnt] = nodeHandle_.advertise<darknet_ros_msgs::BoundingBoxes>("/" + robot_name + boundingBoxesTopicName, boundingBoxesQueueSize, boundingBoxesLatch);
    detectionImagePublishers_[cnt] = nodeHandle_.advertise<sensor_msgs::Image>("/" + robot_name + detectionImageTopicName, detectionImageQueueSize, detectionImageLatch);
    objectPublishers_[cnt] = nodeHandle_.advertise<darknet_ros_msgs::ObjectCount>("/" + robot_name + objectDetectorTopicName, objectDetectorQueueSize, objectDetectorLatch);
    cnt++;
  }
  for (int i = 0; i < nr_haulers; i++) {
    std::string robot_name = "small_hauler_" + boost::lexical_cast<std::string>(i+1);
    imageSubscribers_[cnt]= imageTransport_.subscribe("/" + robot_name + cameraTopicName, cameraQueueSize, boost::bind(&YoloObjectDetector::cameraCallback, this, _1, cnt));
    boundingBoxesPublishers_[cnt] = nodeHandle_.advertise<darknet_ros_msgs::BoundingBoxes>("/" + robot_name + boundingBoxesTopicName, boundingBoxesQueueSize, boundingBoxesLatch);
    detectionImagePublishers_[cnt] = nodeHandle_.advertise<sensor_msgs::Image>("/" + robot_name + detectionImageTopicName, detectionImageQueueSize, detectionImageLatch);
    objectPublishers_[cnt] = nodeHandle_.advertise<darknet_ros_msgs::ObjectCount>("/" + robot_name + objectDetectorTopicName, objectDetectorQueueSize, objectDetectorLatch);
    cnt++;
  }



}

void YoloObjectDetector::cameraCallback(const sensor_msgs::ImageConstPtr& msg, int image_stream_index)
{
  ROS_DEBUG("[YoloObjectDetector] USB image received.");

  cv_bridge::CvImagePtr cam_image;

  try {
    cam_image = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  if (cam_image) {
    {
      boost::unique_lock<boost::shared_mutex> lockImageCallback(mutexImageCallback_);
      imageHeaders_[image_stream_index] = msg->header;
      camImageCopies_[image_stream_index] = cam_image->image.clone();
    }
    {
      boost::unique_lock<boost::shared_mutex> lockImageStatus(mutexImageStatus_);
      imageStatus_[image_stream_index] = true;
    }
    // image sizes always the same across all the sources, can share one set of variables
    frameWidth_ = cam_image->image.size().width;
    frameHeight_ = cam_image->image.size().height;
  }
  return;
}


bool YoloObjectDetector::publishDetectionImage(const cv::Mat& detectionImage, int image_stream_index)
{
  if (detectionImagePublishers_[image_stream_index].getNumSubscribers() < 1)
    return false;
  cv_bridge::CvImage cvImage;
  cvImage.header.stamp = ros::Time::now();
  cvImage.header.frame_id = "detection_image";
  cvImage.encoding = sensor_msgs::image_encodings::BGR8;
  cvImage.image = detectionImage;
  detectionImagePublishers_[image_stream_index].publish(*cvImage.toImageMsg());
  ROS_DEBUG("Detection image has been published.");
  return true;
}

// double YoloObjectDetector::getWallTime()
// {
//   struct timeval time;
//   if (gettimeofday(&time, NULL)) {
//     return 0;
//   }
//   return (double) time.tv_sec + (double) time.tv_usec * .000001;
// }

int YoloObjectDetector::sizeNetwork(network *net)
{
  int i;
  int count = 0;
  for(i = 0; i < net->n; ++i){
    layer l = net->layers[i];
    if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
      count += l.outputs;
    }
  }
  return count;
}

void YoloObjectDetector::rememberNetwork(network *net)
{
  int i;
  int count = 0;
  for(i = 0; i < net->n; ++i){
    layer l = net->layers[i];
    if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
      memcpy(predictions_[demoIndex_] + count, net->layers[i].output, sizeof(float) * l.outputs);
      count += l.outputs;
    }
  }
}

detection *YoloObjectDetector::avgPredictions(network *net, int *nboxes)
{
  int i, j;
  int count = 0;
  fill_cpu(demoTotal_, 0, avg_, 1);
  for(j = 0; j < demoFrame_; ++j){
    axpy_cpu(demoTotal_, 1./demoFrame_, predictions_[j], 1, avg_, 1);
  }
  for(i = 0; i < net->n; ++i){
    layer l = net->layers[i];
    if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
      memcpy(l.output, avg_ + count, sizeof(float) * l.outputs);
      count += l.outputs;
    }
  }
  detection *dets = get_network_boxes(net, buff_[0].w, buff_[0].h, demoThresh_, demoHier_, 0, 1, nboxes);
  return dets;
}

void *YoloObjectDetector::detectInThread()
{
  running_ = 1;
  float nms = .4;

  layer l = net_->layers[net_->n - 1];
  float *X = buffLetter_[(buffIndex_ + 0) % 3].data;
  float *prediction = network_predict(net_, X);

  rememberNetwork(net_);
  detection *dets = 0;
  int nboxes = 0;
  dets = avgPredictions(net_, &nboxes);

  if (nms > 0) do_nms_obj(dets, nboxes, l.classes, nms);

  image display = buff_[(buffIndex_+0) % 3];
  draw_detections(display, dets, nboxes, demoThresh_, demoNames_, demoAlphabet_, demoClasses_);

  // extract the bounding boxes and send them to ROS
  int i, j;
  int count = 0;
  for (i = 0; i < nboxes; ++i) {
    float xmin = dets[i].bbox.x - dets[i].bbox.w / 2.;
    float xmax = dets[i].bbox.x + dets[i].bbox.w / 2.;
    float ymin = dets[i].bbox.y - dets[i].bbox.h / 2.;
    float ymax = dets[i].bbox.y + dets[i].bbox.h / 2.;

    if (xmin < 0)
      xmin = 0;
    if (ymin < 0)
      ymin = 0;
    if (xmax > 1)
      xmax = 1;
    if (ymax > 1)
      ymax = 1;

    // iterate through possible boxes and collect the bounding boxes
    for (j = 0; j < demoClasses_; ++j) {
      if (dets[i].prob[j]) {
        float x_center = (xmin + xmax) / 2;
        float y_center = (ymin + ymax) / 2;
        float BoundingBox_width = xmax - xmin;
        float BoundingBox_height = ymax - ymin;

        // define bounding box
        // BoundingBox must be 1% size of frame (3.2x2.4 pixels)
        if (BoundingBox_width > 0.01 && BoundingBox_height > 0.01) {
          roiBoxes_[count].x = x_center;
          roiBoxes_[count].y = y_center;
          roiBoxes_[count].w = BoundingBox_width;
          roiBoxes_[count].h = BoundingBox_height;
          roiBoxes_[count].Class = j;
          roiBoxes_[count].prob = dets[i].prob[j];
          count++;
        }
      }
    }
  }

  // create array to store found bounding boxes
  // if no object detected, make sure that ROS knows that num = 0
  if (count == 0) {
    roiBoxes_[0].num = 0;
  } else {
    roiBoxes_[0].num = count;
  }

  free_detections(dets, nboxes);
  demoIndex_ = (demoIndex_ + 1) % demoFrame_;
  running_ = 0;
  return 0;
}

void *YoloObjectDetector::fetchInThread(int image_stream_index)
{
  {
    boost::shared_lock<boost::shared_mutex> lock(mutexImageCallback_);
    MatWithHeader_ imageAndHeader = getIplImageWithHeader(image_stream_index);
    cv::Mat ROS_img = imageAndHeader.image;
    mat_to_image(ROS_img, buff_ + buffIndex_);
    headerBuff_[buffIndex_] = imageAndHeader.header;
    buffId_[buffIndex_] = actionId_;
  }
  rgbgr_image(buff_[buffIndex_]);
  letterbox_image_into(buff_[buffIndex_], net_->w, net_->h, buffLetter_[buffIndex_]);
  return 0;
}

void YoloObjectDetector::setupNetwork(char *cfgfile, char *weightfile, char *datafile, float thresh,
                                      char **names, int classes,
                                      int delay, char *prefix, int avg_frames, float hier, int w, int h,
                                      int frames, int fullscreen)
{
  demoPrefix_ = prefix;
  demoDelay_ = delay;
  demoFrame_ = avg_frames;
  image **alphabet = load_alphabet_with_file(datafile);
  demoNames_ = names;
  demoAlphabet_ = alphabet;
  demoClasses_ = classes;
  demoThresh_ = thresh;
  demoHier_ = hier;
  fullScreen_ = fullscreen;
  printf("YOLO V3\n");
  net_ = load_network(cfgfile, weightfile, 0);
  set_batch_network(net_, 1);
}

void YoloObjectDetector::yolo()
{
  const auto wait_duration = std::chrono::milliseconds(2000);

  ROS_INFO("[YoloObjectDetector] rate=%d Hz", publishRate_);
  ros::Rate rate(publishRate_);

  while (true) {
    bool ready = true;
    for (int image_stream_index = 0; image_stream_index < (nr_scouts + nr_excavators + nr_haulers); image_stream_index++) {
      ready = ready && getImageStatus(image_stream_index);
    }
    if (ready)
      break;

    printf("Waiting for image(s).\n");
    if (!isNodeRunning()) {
      return;
    }
    std::this_thread::sleep_for(wait_duration);
  }

  srand(2222222);

  int i;
  demoTotal_ = sizeNetwork(net_);
  predictions_ = (float **) calloc(demoFrame_, sizeof(float*));
  for (i = 0; i < demoFrame_; ++i){
      predictions_[i] = (float *) calloc(demoTotal_, sizeof(float));
  }
  avg_ = (float *) calloc(demoTotal_, sizeof(float));

  layer l = net_->layers[net_->n - 1];
  roiBoxes_ = (darknet_ros::RosBox_ *) calloc(l.w * l.h * l.n, sizeof(darknet_ros::RosBox_));

  {
    boost::shared_lock<boost::shared_mutex> lock(mutexImageCallback_);
    MatWithHeader_ imageAndHeader = getIplImageWithHeader(0); // grab image from first stream to populate buffers
    cv::Mat ROS_img = imageAndHeader.image;
    buff_[0] = mat_to_image(ROS_img);
    headerBuff_[0] = imageAndHeader.header;
  }
  buff_[1] = copy_image(buff_[0]);
  buff_[2] = copy_image(buff_[0]);
  headerBuff_[1] = headerBuff_[0];
  headerBuff_[2] = headerBuff_[0];
  buffLetter_[0] = letterbox_image(buff_[0], net_->w, net_->h);
  buffLetter_[1] = letterbox_image(buff_[0], net_->w, net_->h);
  buffLetter_[2] = letterbox_image(buff_[0], net_->w, net_->h);
  ipl_ = cv::Mat(buff_[0].h, buff_[0].w,
          CV_8UC(buff_[0].c));

  int count = 0;

  while (!demoDone_) {
    for (int image_stream_index = 0; image_stream_index < nr_excavators + nr_haulers + nr_scouts; image_stream_index++) {
      fetchInThread(image_stream_index);
      detectInThread();
      generate_image(buff_[(buffIndex_ + 0)%3], ipl_);
      publishInThread(image_stream_index);
      ++count;
      if (!isNodeRunning()) {
        demoDone_ = true;
      }
    }
    rate.sleep();
  }

}

MatWithHeader_ YoloObjectDetector::getIplImageWithHeader(int image_stream_index)
{
  MatWithHeader_ header {camImageCopies_[image_stream_index], imageHeaders_[image_stream_index]};
  return header;
}

bool YoloObjectDetector::getImageStatus(int image_stream_index)
{
  boost::shared_lock<boost::shared_mutex> lock(mutexImageStatus_);
  return imageStatus_[image_stream_index];
}

bool YoloObjectDetector::isNodeRunning(void)
{
  boost::shared_lock<boost::shared_mutex> lock(mutexNodeStatus_);
  return isNodeRunning_;
}

void *YoloObjectDetector::publishInThread(int image_stream_index)
{
  // Publish image.
  cv::Mat cvImage = ipl_;
  if (!publishDetectionImage(cv::Mat(cvImage), image_stream_index)) {
    ROS_DEBUG("Detection image has not been broadcasted.");
  }

  // Publish bounding boxes and detection result.
  int num = roiBoxes_[0].num;
  if (num >= 0 && num <= 100) {
    for (int i = 0; i < num; i++) {
      for (int j = 0; j < numClasses_; j++) {
        if (roiBoxes_[i].Class == j) {
          rosBoxes_[j].push_back(roiBoxes_[i]);
          rosBoxCounter_[j]++;
        }
      }
    }

    darknet_ros_msgs::ObjectCount msg;
    msg.header.stamp = ros::Time::now();
    msg.header.frame_id = "detection";
    msg.count = num;
    objectPublishers_[image_stream_index].publish(msg);

    for (int i = 0; i < numClasses_; i++) {
      if (rosBoxCounter_[i] > 0) {
        darknet_ros_msgs::BoundingBox boundingBox;

        for (int j = 0; j < rosBoxCounter_[i]; j++) {
          int xmin = (rosBoxes_[i][j].x - rosBoxes_[i][j].w / 2) * frameWidth_;
          int ymin = (rosBoxes_[i][j].y - rosBoxes_[i][j].h / 2) * frameHeight_;
          int xmax = (rosBoxes_[i][j].x + rosBoxes_[i][j].w / 2) * frameWidth_;
          int ymax = (rosBoxes_[i][j].y + rosBoxes_[i][j].h / 2) * frameHeight_;

          boundingBox.Class = classLabels_[i];
          boundingBox.id = i;
          boundingBox.probability = rosBoxes_[i][j].prob;
          boundingBox.xmin = xmin;
          boundingBox.ymin = ymin;
          boundingBox.xmax = xmax;
          boundingBox.ymax = ymax;
          boundingBoxesResults_.bounding_boxes.push_back(boundingBox);
        }
      }
    }
    boundingBoxesResults_.header.stamp = ros::Time::now();
    boundingBoxesResults_.header.frame_id = "detection";
    boundingBoxesResults_.image_header = headerBuff_[(buffIndex_ + 0) % 3];
    boundingBoxesPublishers_[image_stream_index].publish(boundingBoxesResults_);
  } else {
    darknet_ros_msgs::ObjectCount msg;
    msg.header.stamp = ros::Time::now();
    msg.header.frame_id = "detection";
    msg.count = 0;
    objectPublishers_[image_stream_index].publish(msg);
  }
  boundingBoxesResults_.bounding_boxes.clear();
  for (int i = 0; i < numClasses_; i++) {
    rosBoxes_[i].clear();
    rosBoxCounter_[i] = 0;
  }

  return 0;
}


} /* namespace darknet_ros*/
