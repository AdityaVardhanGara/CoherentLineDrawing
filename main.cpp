/*

Main function:-
taking input and performing alal the steps one by one 
Parameters:-
halfw	smoothpasses	sigma1		sigma2		tau
*/
#include "opencv2/opencv.hpp"

#include "imatrix.h"
#include "ETF.h"
#include "fdog.h"
#include "myvec.h"

using cv::Mat;


// coherent line drawing: good values for halfw are between 1 and 8,
// smoothPasses 1, and 4, sigma1 between .01 and 2, sigma2 between .01 and 10,
// tau between .8 and 1.0
// this could be rewritten into a class so we're not doing an allocate and copy each time
/*
parameters:
halw= degree of coherence ()
smooth passes=line width
sigma1=sigma m
sigma2=sigma c
tau=thresholding
*/
void CLD(Mat src, Mat &dst, int halfw = 4, int smoothPasses = 2, double sigma1 = .4, double sigma2 = 3, double tau = .99, int black = 0) {
    src.copyTo(dst);
    int width = src.cols, height = src.rows;
    imatrix img;
    //creating  a imatrix img of source size
    img.init(height, width);
    if(black != 0) {
        add(dst, cv::Scalar(black), dst);
    }
    // copy from dst (unsigned char) to img (int)
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            img[y][x] = dst.at<unsigned char>(y, x);
        }
    }
    ETF etf;
    etf.init(height, width);
    etf.set(img);
    //smooth the image obtained with smoothing parameters we declared.. halfw=4 and and smoothpasses=2
    etf.Smooth(halfw, smoothPasses);
    //Flow-based Difference-of-Gaussians
    GetFDoG(img, etf, sigma1, sigma2, tau);
    // copy result from img (int) to dst (unsigned char)
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            dst.at<unsigned char>(y, x) = img[y][x];
        }
    }
}

int main(int argc, char *argv[]) {
    argc--;
    argv++;
    //error message incase of less number of parameters are passed
    if (argc < 2) {
        fprintf(stderr, "main in.png out.png\n"); exit(1);
    }
    //storing image as matrix form
    Mat image;
    image = cv::imread(argv[0], cv::IMREAD_GRAYSCALE);

    Mat out_image;
    //Extract coherent lines from the image
    CLD(image, out_image);

    cv::imwrite(argv[1], out_image);
    return 0;
}
