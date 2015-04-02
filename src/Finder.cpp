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

#include "Finder.h"

#include <opencv2/imgproc/imgproc.hpp>


//==================================================================================================
// C O N S T A N T S   &   L O C A L   V A R I A B L E S

//==================================================================================================
// G L O B A L S

//==================================================================================================
// C O N S T R U C T O R (S) / D E S T R U C T O R   C O D E   S E C T I O N

//--------------------------------------------------------------------------------------------------
//
ObjectFinder::ObjectFinder()
	: threshold_( 0.1f )
	, isSparse_( false )
{
	ranges_[0] = hranges_; // all channels have the same range
	ranges_[1] = hranges_;
	ranges_[2] = hranges_;
}

//--------------------------------------------------------------------------------------------------
//
ObjectFinder::~ObjectFinder()
{ }


//=================================================================================================
// M E T H O D S   C O D E   S E C T I O N

//--------------------------------------------------------------------------------------------------
//
void
ObjectFinder::set_threshold( const float t )
{
	threshold_ = t;
}

//--------------------------------------------------------------------------------------------------
//
float
ObjectFinder::get_threshold()
{
	return threshold_;
}

//--------------------------------------------------------------------------------------------------
//
void
ObjectFinder::set_histogram( const cv::MatND& h )
{
	isSparse_ = false;
	histogram_ = h;
	cv::normalize( histogram_, histogram_, 1.0 );
}

//--------------------------------------------------------------------------------------------------
//
void
ObjectFinder::set_histogram( const cv::SparseMat& h )
{
	isSparse_ = true;
	shistogram_ = h;
	cv::normalize( shistogram_, shistogram_, 1.0, cv::NORM_L2 );
}

//--------------------------------------------------------------------------------------------------
//
cv::Mat
ObjectFinder::find( const cv::Mat& image )
{
	cv::Mat result;

	hranges_[0] = 0.0;    // range [0,255]
	hranges_[1] = 255.0;
	channels_[0] = 0;     // the three channels
	channels_[1] = 1;
	channels_[2] = 2;

	if( isSparse_ ) // call the right function based on histogram type
	{

		cv::calcBackProject( &image,
		                     1,            // one image
		                     channels_,     // vector specifying what histogram dimensions belong to what image channels
		                     shistogram_,   // the histogram we are using
		                     result,       // the resulting back projection image
		                     ranges_,       // the range of values, for each dimension
		                     255.0         // the scaling factor is chosen such that a histogram value of 1 maps to 255
		                   );
	}
	else
	{
		cv::calcBackProject( &image,
		                     1,            // one image
		                     channels_,     // vector specifying what histogram dimensions belong to what image channels
		                     histogram_,    // the histogram we are using
		                     result,       // the resulting back projection image
		                     ranges_,       // the range of values, for each dimension
		                     255.0         // the scaling factor is chosen such that a histogram value of 1 maps to 255
		                   );
	}

	// Threshold back projection to obtain a binary image
	if( threshold_ > 0.0 )
	{
		cv::threshold( result, result, 255 * threshold_, 255, cv::THRESH_BINARY );
	}

	return result;
}

//--------------------------------------------------------------------------------------------------
//
cv::Mat
ObjectFinder::find( const cv::Mat& image, float minValue, float maxValue, int* channels, int dim )
{
	cv::Mat result;

	hranges_[0] = minValue;
	hranges_[1] = maxValue;

	for( int i = 0; i < dim; ++i )
	{
		this->channels_[i] = channels[i];
	}

	if( isSparse_ ) // call the right function based on histogram type
	{
		cv::calcBackProject( &image,
		                     1,            // we only use one image at a time
		                     channels_,     // vector specifying what histogram dimensions belong to what image channels
		                     shistogram_,   // the histogram we are using
		                     result,       // the resulting back projection image
		                     ranges_,       // the range of values, for each dimension
		                     255.0         // the scaling factor is chosen such that a histogram value of 1 maps to 255
		                   );

	}
	else
	{
		cv::calcBackProject( &image,
		                     1,            // we only use one image at a time
		                     channels_,    // vector specifying what histogram dimensions belong to what image channels
		                     histogram_,   // the histogram we are using
		                     result,       // the resulting back projection image
		                     ranges_,      // the range of values, for each dimension
		                     255.0         // the scaling factor is chosen such that a histogram value of 1 maps to 255
		                   );
	}

	// Threshold back projection to obtain a binary image
	if( threshold_ > 0.0 )
	{
		cv::threshold( result, result, 255 * threshold_, 255, cv::THRESH_BINARY );
	}

	return result;
}
