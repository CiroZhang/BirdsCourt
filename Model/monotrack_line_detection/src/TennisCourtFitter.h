//
// Created by Chlebus, Grzegorz on 28.08.17.
// Copyright (c) Chlebus, Grzegorz. All rights reserved.
//
#pragma once

#include "Line.h"
#include <opencv2/opencv.hpp>
#include "TennisCourtModel.h"

class TennisCourtFitter
{
public:
  struct Parameters
  {
    Parameters();
  };

  TennisCourtFitter();

  TennisCourtFitter(Parameters p);

  TennisCourtModel run(const std::vector<Line>& lines, const cv::Mat& binaryImage, const cv::Mat& rgbImage);

  // Every plausible candidate considered during the search, not just the
  // single winner -- kept so a learned re-ranker can be trained on which
  // candidate was actually right, instead of trusting the blind pixel-count
  // argmax. Only candidates that passed evaluateModel's basic sanity checks
  // (non-degenerate, convex, not mostly off-screen) are kept -- score >
  // GlobalParameters().initialFitScore is exactly that filter.
  //
  // Net is deliberately excluded here: fitNet()'s independent line search is
  // unreliable (frequently fails outright) and was previously folded into
  // the ranking score, which actively corrupted COURT candidate selection
  // (confirmed: in ~8% of images, the best-scoring candidate by court score
  // alone was a meaningfully better court fit than whatever won once net
  // noise was mixed in). Net position is handled entirely by downstream
  // geometric reprojection against the solved homography instead.
  struct Candidate
  {
    TennisCourtModel model;
    float score;
  };
  std::vector<Candidate> allCandidates;

  static bool debug;
  static const std::string windowName;

private:
  void getHorizontalAndVerticalLines(const std::vector<Line>& lines, std::vector<Line>& hLines,
    std::vector<Line>& vLines, const cv::Mat& rgbImage, int mode=1);

  void sortHorizontalLines(std::vector<Line>& hLines, const cv::Mat& rgbImage);

  void sortVerticalLines(std::vector<Line>& vLines, const cv::Mat& rgbImage);

  void findBestModelFit(const std::vector<Line>& lines, const cv::Mat& binaryImage, const cv::Mat& rgbImage, int mode);

  Parameters parameters;
  std::vector<LinePair> hLinePairs;
  std::vector<LinePair> vLinePairs;
  TennisCourtModel bestModel;
  float bestScore;
};