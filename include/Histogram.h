//==================================================================================================
//
//  Copyright(c)  2013  Jean Inderchit
//
//  This program is free software: you can redistribute it and/or modify it under the terms of the GNU
//	General Public License as published by the Free Software Foundation, either version 3 of the
//	License, or (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
//  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//  See the GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License along with This program.
//  If not, see <http://www.gnu.org/licenses/>.
//
//==================================================================================================

#ifndef HISTOGRAM_H
#define HISTOGRAM_H

//==================================================================================================
// I N C L U D E   F I L E S   A N D   F O R W A R D   D E C L A R A T I O N S

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>


//==================================================================================================
// C O N S T A N T S

//==================================================================================================
// C L A S S E S


class Histogram1D
{
//--Methods-----------------------------------------------------------------------------------------
public:
	Histogram1D();

	~Histogram1D() = default;

	// Sets the channel on which histogram will be calculated.
	// By default it is channel 0.
	void set_channel( const int32_t c );

	// Gets the channel used.
	int32_t get_channel();

	// Sets the range for the pixel values.
	// By default it is [0,255]
	void setRange( const float minValue, const float maxValue );

	// Gets the min pixel value.
	float getMinValue();

	// Gets the max pixel value.
	float getMaxValue();

	// Sets the number of bins in histogram.
	// By default it is 256.
	void setNBins( const int32_t nbins );

	// Gets the number of bins in histogram.
	int32_t getNBins();

	// Computes the 1D histogram.
	cv::MatND getHistogram( const cv::Mat& image );

	// Computes the 1D histogram and returns an image of it.
	cv::Mat getHistogramImage( const cv::Mat& image );

	// Equalizes the source image.
	cv::Mat equalize( const cv::Mat& image );

	// Stretches the source image.
	cv::Mat stretch( const cv::Mat& image, int32_t minValue = 0 );

	// Applies a lookup table transforming an input image into a 1-channel image
	cv::Mat applyLookUp( const cv::Mat& image, const cv::MatND& lookup );

private:
	int32_t histSize_[1];
	float hranges_[2];
	const float* ranges_[1];
	int32_t channels_[1];
};

class ColorHistogram
{
public:
	ColorHistogram();

	~ColorHistogram() = default;

	// Computes the histogram.
	cv::MatND get_histogram( const cv::Mat& image );

	// Computes the histogram.
	cv::SparseMat get_sparse_histogram( const cv::Mat& image );

	// Computes the 2D ab histogram.
	// BGR source image is converted to Lab
	cv::MatND get_ab_histogram( const cv::Mat& image );

	// Computes the 1D Hue histogram with a mask.
	// BGR source image is converted to HSV
	cv::MatND get_hue_histogram( const cv::Mat& image );

	cv::Mat color_reduce( const cv::Mat& image, int32_t div = 64 );

private:
	int32_t histSize_[3];
	float hranges_[2];
	const float* ranges_[3];
	int32_t channels_[3];
};

//==================================================================================================
// I N L I N E   F U N C T I O N S   C O D E   S E C T I O N

#endif
