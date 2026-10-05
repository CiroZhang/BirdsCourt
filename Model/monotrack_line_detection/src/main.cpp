#include <opencv2/opencv.hpp>
#include <algorithm>
#include <fstream>
#include <sys/stat.h>

#include "TimeMeasurement.h"
#include "Line.h"
#include "CourtLinePixelDetector.h"
#include "CourtLineCandidateDetector.h"
#include "TennisCourtFitter.h"
#include "DebugHelpers.h"

using namespace cv;

int main(int argc, char** argv)
{
  TimeMeasurement::debug = false;
  CourtLinePixelDetector::debug = false;
  CourtLineCandidateDetector::debug = false;
  TennisCourtFitter::debug = false;

  if (argc < 2 || argc > 5)
  {
    std::cout << "Usage: ./detect video_path [output_path] [output_image_path] [candidates_dir]" << std::endl;
    std::cout << "       video_path:  path to an input avi file." << std::endl;
    std::cout << "       output_path: path to an output file where the xy court point coordinates will be written." << std::endl;
    std::cout << "                    This argument is optional. If not present, then a window with the result will be opened." << std::endl;
    std::cout << "       output_image_path: path to an output file where the image will be written." << std::endl;
    std::cout << "                    This argument is optional. If not present, then a window with the result will be opened." << std::endl;
    std::cout << "       candidates_dir: directory to dump EVERY plausible candidate fit (not just the" << std::endl;
    std::cout << "                    winner) as cand_<i>.pts + a scores.csv -- for training a learned" << std::endl;
    std::cout << "                    re-ranker. Optional; omit to keep the original behavior." << std::endl;
    return -1;
  }
  std::string filename(argv[1]);

  std::cout << "Reading file " << filename << std::endl;
  VideoCapture vc(filename);
  if (!vc.isOpened())
  {
    std::cerr << "Cannot open file " << filename << std::endl;
    return 1;
  }
  printVideoInfo(vc);
  Mat frame;
  int frameIndex = int(vc.get(CAP_PROP_FRAME_COUNT)) / 2;
  vc.set(CAP_PROP_POS_FRAMES, frameIndex);
  if (!vc.read(frame))
  {
    std::cerr << "Failed to read frame with index " << frameIndex << std::endl;
    return 2;
  }
  std::cout << "Reading frame with index " << frameIndex << std::endl;

  CourtLinePixelDetector courtLinePixelDetector;
  CourtLineCandidateDetector courtLineCandidateDetector;
  TennisCourtFitter tennisCourtFitter;

  std::cout << "Starting court line detection algorithm..." << std::endl;
  try
  {
    TimeMeasurement::start("LineDetection");
    Mat binaryImage = courtLinePixelDetector.run(frame);
    if (const char* dbg = std::getenv("MONOTRACK_DEBUG_BINARY")) {
      writeImage(dbg, binaryImage);
    }
    std::vector<Line> candidateLines = courtLineCandidateDetector.run(binaryImage, frame);
    TennisCourtModel model = tennisCourtFitter.run(candidateLines, binaryImage, frame);
    int elapsed_seconds = TimeMeasurement::stop("LineDetection");
    std::cout << "Elapsed time: " << elapsed_seconds << "s." << std::endl;
    if (argc == 2)
    {
      model.drawModel(frame);
      displayImage("Result - press key to exit", frame);
    }
    if (argc >= 3)
    {
      std::string outFilename(argv[2]);
      model.writeToFile(outFilename);
      std::cout << "Result written to " << outFilename << std::endl;
    }
    if (argc >= 4)
    {
      std::string outFilename(argv[3]);
      model.drawModel(frame);
      writeImage(outFilename, frame);
    }
    if (argc >= 5)
    {
      std::string candidatesDir(argv[4]);
      mkdir(candidatesDir.c_str(), 0755);

      // The search (especially its "random mode" fallback) evaluates
      // thousands of line-pair combinations per image, most of them nowhere
      // close to plausible. A re-ranker only needs the handful that actually
      // looked reasonable, so keep just the top-K by COURT score alone --
      // net is excluded from this ranking entirely (see TennisCourtFitter.h),
      // so a genuinely good court fit is never crowded out of the pool just
      // because fitNet() would have failed to find a net.
      const size_t TOP_K = 30;
      auto candidates = tennisCourtFitter.allCandidates;  // copy, sort doesn't disturb the original
      std::sort(candidates.begin(), candidates.end(),
                [](const TennisCourtFitter::Candidate& a, const TennisCourtFitter::Candidate& b) {
                  return a.score > b.score;
                });
      size_t keep = std::min(TOP_K, candidates.size());

      std::ofstream scoresCsv(candidatesDir + "/scores.csv");
      scoresCsv << "idx,score\n";
      for (size_t i = 0; i < keep; i++)
      {
        auto& cand = candidates[i];
        std::string candPath = candidatesDir + "/cand_" + std::to_string(i) + ".pts";
        cand.model.writeToFile(candPath);
        scoresCsv << i << "," << cand.score << "\n";
      }
      std::cout << "Wrote top " << keep << " of " << tennisCourtFitter.allCandidates.size()
                << " candidates considered to " << candidatesDir << std::endl;
    }

  }
  catch (std::runtime_error& e)
  {
    std::cout << "Processing error: " << e.what() << std::endl;
    return 3;
  }


  return 0;
}
