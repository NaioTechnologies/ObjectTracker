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


//==================================================================================================
// I N C L U D E   F I L E S

#include "Histogram.h"


//==================================================================================================
// C O N S T A N T S   &   L O C A L   V A R I A B L E S

//==================================================================================================
// G L O B A L S

//==================================================================================================
// C O N S T R U C T O R (S) / D E S T R U C T O R   C O D E   S E C T I O N

//--------------------------------------------------------------------------------------------------
//
Histogram1D::Histogram1D()
{
	// Prepare arguments for 1D histogram
	histSize_[0] = 256;
	hranges_[0] = 0.0;
	hranges_[1] = 255.0;
	ranges_[0] = hranges_;
	channels_[0] = 0; // by default, we look at channel 0
}

//=================================================================================================
// M E T H O D S   C O D E   S E C T I O N

//--------------------------------------------------------------------------------------------------
//
void
Histogram1D::set_channel( const int c )
{
	channels_[0] = c;
}

//--------------------------------------------------------------------------------------------------
//
int32_t
Histogram1D::get_channel()
{
	return channels_[0];
}

//--------------------------------------------------------------------------------------------------
//
void
Histogram1D::setRange( const float minValue, const float maxValue )
{
	hranges_[0] = minValue;
	hranges_[1] = maxValue;
}

//--------------------------------------------------------------------------------------------------
//
float
Histogram1D::getMinValue()
{
	return hranges_[0];
}

//--------------------------------------------------------------------------------------------------
//
float
Histogram1D::getMaxValue()
{
	return hranges_[1];
}

//--------------------------------------------------------------------------------------------------
//
void
Histogram1D::setNBins( const int32_t nbins )
{
	histSize_[0] = nbins;
}

//--------------------------------------------------------------------------------------------------
//
int32_t
Histogram1D::getNBins()
{
	return histSize_[0];
}

//--------------------------------------------------------------------------------------------------
//
cv::MatND
Histogram1D::getHistogram( const cv::Mat& image )
{
	cv::MatND hist;

	// Compute histogram
	cv::calcHist( &image,
	              1,          // histogram of 1 image only
	              channels_,  // the channel used
	              cv::Mat(),  // no mask is used
	              hist,       // the resulting histogram
	              1,          // it is a 1D histogram
	              histSize_,  // number of bins
	              ranges_ );  // pixel value range

	return hist;
}

//--------------------------------------------------------------------------------------------------
//
cv::Mat
Histogram1D::getHistogramImage( const cv::Mat& image )
{
	// Compute histogram first
	cv::MatND hist = getHistogram( image );

	// Get min and max bin values
	double maxVal{ };
	double minVal{ };

	cv::minMaxLoc( hist, &minVal, &maxVal, 0, 0 );

	// Image on which to display histogram
	cv::Mat histImg( histSize_[0], histSize_[0], CV_8U, cv::Scalar( 255 ) );

	// set highest point at 90% of nbins
	int hpt = static_cast<int>( 0.9 * histSize_[0] );

	// Draw vertical line for each bin
	for( int h = 0; h < histSize_[0]; ++h )
	{
		float binVal = hist.at< float >( h );
		int intensity = static_cast<int>( binVal * hpt / maxVal );

		cv::line( histImg, cv::Point( h, histSize_[0] ),
		          cv::Point( h, histSize_[0] - intensity ), cv::Scalar::all( 0 ) );
	}

	return histImg;
}

//--------------------------------------------------------------------------------------------------
//
cv::Mat
Histogram1D::equalize( const cv::Mat& image )
{
	cv::Mat result;
	cv::equalizeHist( image, result );

	return result;
}

//--------------------------------------------------------------------------------------------------
//
cv::Mat
Histogram1D::stretch( const cv::Mat& image, int32_t minValue )
{
	// Compute histogram first
	cv::MatND hist = getHistogram( image );

	// find left extremity of the histogram
	int imin = 0;

	for( ; imin < histSize_[0]; imin++ )
	{
		std::cout << hist.at< float >( imin ) << std::endl;

		if( hist.at< float >( imin ) > minValue )
		{
			break;
		}
	}

	// find right extremity of the histogram
	int imax = histSize_[0] - 1;
	for( ; imax >= 0; imax-- )
	{
		if( hist.at< float >( imax ) > minValue )
		{
			break;
		}
	}

	// Create lookup table
	int dims[1] = { 256 };
	cv::MatND lookup( 1, dims, CV_8U );

	for( int i = 0; i < 256; ++i )
	{
		if( i < imin )
		{
			lookup.at< uchar >( i ) = 0;
		}
		else if( i > imax )
		{
			lookup.at< uchar >( i ) = 255;
		}
		else
		{
			lookup.at< uchar >( i ) = static_cast<uchar>( 255.0 * (i - imin) /
			                                              (imax - imin) + 0.5 );
		}
	}

	// Apply lookup table
	cv::Mat result;
	result = applyLookUp( image, lookup );

	return result;
}

//--------------------------------------------------------------------------------------------------
//
cv::Mat
Histogram1D::applyLookUp( const cv::Mat& image, const cv::MatND& lookup )
{
	// Set output image (always 1-channel)
	cv::Mat result( image.rows, image.cols, CV_8U );
	cv::Mat_< uchar >::iterator itr = result.begin< uchar >();

	// Iterates over the input image
	cv::Mat_< uchar >::const_iterator it = image.begin< uchar >();
	cv::Mat_< uchar >::const_iterator itend = image.end< uchar >();

	// Applies lookup to each pixel
	for( ; it != itend; ++it, ++itr )
	{
		*itr = lookup.at< uchar >( *it );
	}

	return result;
}

//==================================================================================================
// C O N S T R U C T O R (S) / D E S T R U C T O R   C O D E   S E C T I O N

//--------------------------------------------------------------------------------------------------
//
ColorHistogram::ColorHistogram()
{
	// Prepare arguments for a color histogram
	histSize_[0] = histSize_[1] = histSize_[2] = 256;
	hranges_[0] = 0.0;    // BRG range
	hranges_[1] = 255.0;
	ranges_[0] = hranges_; // all channels have the same range
	ranges_[1] = hranges_;
	ranges_[2] = hranges_;
	channels_[0] = 0;     // the three channels
	channels_[1] = 1;
	channels_[2] = 2;
}

//=================================================================================================
// M E T H O D S   C O D E   S E C T I O N

//--------------------------------------------------------------------------------------------------
//
cv::MatND
ColorHistogram::get_histogram( const cv::Mat& image )
{
	cv::MatND hist;

	// BGR color histogram
	hranges_[0] = 0.0;    // BRG range
	hranges_[1] = 255.0;
	channels_[0] = 0;     // the three channels
	channels_[1] = 1;
	channels_[2] = 2;

	// Compute histogram
	cv::calcHist( &image,
	              1,          // histogram of 1 image only
	              channels_,   // the channel used
	              cv::Mat(),  // no mask is used
	              hist,       // the resulting histogram
	              3,          // it is a 3D histogram
	              histSize_,   // number of bins
	              ranges_      // pixel value range
	            );

	return hist;
}

//--------------------------------------------------------------------------------------------------
//
cv::SparseMat
ColorHistogram::get_sparse_histogram( const cv::Mat& image )
{
	cv::SparseMat hist( 3, histSize_, CV_32F );

	// BGR color histogram
	hranges_[0] = 0.0;    // BRG range
	hranges_[1] = 255.0;
	channels_[0] = 0;     // the three channels
	channels_[1] = 1;
	channels_[2] = 2;

	// Compute histogram
	cv::calcHist( &image,
	              1,          // histogram of 1 image only
	              channels_,   // the channel used
	              cv::Mat(),  // no mask is used
	              hist,       // the resulting histogram
	              3,          // it is a 3D histogram
	              histSize_,   // number of bins
	              ranges_      // pixel value range
	            );

	return hist;
}

//--------------------------------------------------------------------------------------------------
//
cv::MatND
ColorHistogram::get_ab_histogram( const cv::Mat& image )
{
	cv::MatND hist;

	// Convert to Lab color space
	cv::Mat lab;
	cv::cvtColor( image, lab, CV_BGR2Lab );

	// Prepare arguments for a 2D color histogram
	hranges_[0] = 0;
	hranges_[1] = 255;
	channels_[0] = 1; // the two channels used are ab
	channels_[1] = 2;

	// Compute histogram
	cv::calcHist( &lab,
	              1,          // histogram of 1 image only
	              channels_,   // the channel used
	              cv::Mat(),  // no mask is used
	              hist,       // the resulting histogram
	              2,          // it is a 2D histogram
	              histSize_,   // number of bins
	              ranges_      // pixel value range
	            );

	return hist;
}

//--------------------------------------------------------------------------------------------------
//
cv::MatND
ColorHistogram::get_hue_histogram( const cv::Mat& image )
{
	cv::MatND hist;

	// Convert to Lab color space
	cv::Mat hue;
	cv::cvtColor( image, hue, CV_BGR2HSV );

	// Prepare arguments for a 1D hue histogram
	hranges_[0] = 0.0;
	hranges_[1] = 180.0;
	channels_[0] = 0; // the hue channel

	// Compute histogram
	cv::calcHist( &hue,
	              1,          // histogram of 1 image only
	              channels_,   // the channel used
	              cv::Mat(),  // no mask is used
	              hist,       // the resulting histogram
	              1,          // it is a 1D histogram
	              histSize_,   // number of bins
	              ranges_      // pixel value range
	            );

	return hist;
}

//--------------------------------------------------------------------------------------------------
//
cv::Mat
ColorHistogram::color_reduce( const cv::Mat& image, int32_t div )
{
	int32_t n = static_cast<int>( log( static_cast<double>(div) ) / log( 2.0 ) );

	// mask used to round the pixel value
	uchar mask = 0xFF << n; // e.g. for div=16, mask= 0xF0

	cv::Mat_< cv::Vec3b >::const_iterator it = image.begin< cv::Vec3b >();
	cv::Mat_< cv::Vec3b >::const_iterator itend = image.end< cv::Vec3b >();

	// Set output image (always 1-channel)
	cv::Mat result( image.rows, image.cols, image.type() );
	cv::Mat_< cv::Vec3b >::iterator itr = result.begin< cv::Vec3b >();

	for( ; it != itend; ++it, ++itr )
	{
		(*itr)[0] = ((*it)[0] & mask) + div / 2;
		(*itr)[1] = ((*it)[1] & mask) + div / 2;
		(*itr)[2] = ((*it)[2] & mask) + div / 2;
	}

	return result;
}
