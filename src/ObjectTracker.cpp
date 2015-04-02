//=================================================================================================
//
//  Copyright(c)  2013  Jean Inderchit
//
//  Vitals is free software: you can redistribute it and/or modify it under the terms of the GNU
//	General Public License as published by the Free Software Foundation, either version 3 of the
//	License, or (at your option) any later version.
//
//  Vitals is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
//	even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License along with Vitals. If not,
//	see <http://www.gnu.org/licenses/>.
//
//=================================================================================================

//=================================================================================================
// I N C L U D E   F I L E S

#include <CLPrint.h>
#include "CLMath.h"
#include "ObjectTracker.h"


//=================================================================================================
// C O N S T A N T S   &   L O C A L   V A R I A B L E S

//=================================================================================================
// G L O B A L S

//=================================================================================================
// C O N S T R U C T O R (S) / D E S T R U C T O R   C O D E   S E C T I O N

//-------------------------------------------------------------------------------------------------
//
ObjectTracker::ObjectTracker()
	: predicted_position_{ }
	, kalman_filter_{ 4, 2, 0 }
	, measurement_{ 2, 1 }
	, ellipses_{ 50 }
	, has_polygon_{ }
	, lost_count_{ }
	, index_{ }
{ }

//-------------------------------------------------------------------------------------------------
//
ObjectTracker::~ObjectTracker()
{ }


//=================================================================================================
// M E T H O D S   C O D E   S E C T I O N

//-------------------------------------------------------------------------------------------------
//
void
ObjectTracker::initialize( const cv::Point2i& measured_position, uint32_t index )
{
	kalman_filter_.transitionMatrix =
		*(cv::Mat_< float >( 4, 4 ) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1);

	measurement_.setTo( cv::Scalar( 0 ) );

	// Init...
	kalman_filter_.statePre.at< float >( 0 ) = static_cast<float>( measured_position.x );
	kalman_filter_.statePre.at< float >( 1 ) = static_cast<float>( measured_position.y );
	kalman_filter_.statePre.at< float >( 2 ) = 0.0;
	kalman_filter_.statePre.at< float >( 3 ) = 0.0;
	setIdentity( kalman_filter_.measurementMatrix );
	setIdentity( kalman_filter_.processNoiseCov, cv::Scalar::all( 1e-4 ) );
	setIdentity( kalman_filter_.measurementNoiseCov, cv::Scalar::all( 1e-1 ) );
	setIdentity( kalman_filter_.errorCovPost, cv::Scalar::all( .1 ) );

	index_ = index;
}

//-------------------------------------------------------------------------------------------------
//
void
ObjectTracker::compute_prediction()
{
	// First predict, to update the internal statePre variable
	cv::Mat prediction = kalman_filter_.predict();
	predicted_position_ = cv::Point2f( prediction.at< float >( 0 ), prediction.at< float >( 1 ) );
}

//-------------------------------------------------------------------------------------------------
//
cv::Point2i
ObjectTracker::get_predicted_position()
{
	return cv::Point2i{ cl::math::RoundToLong( predicted_position_.x ),
	                    cl::math::RoundToLong( predicted_position_.y ) };
}

//-------------------------------------------------------------------------------------------------
//
cv::Point2i
ObjectTracker::get_estimated_position( const cv::Point2i& measured_position )
{
	// Get mouse point
	measurement_( 0, 0 ) = static_cast<float>( measured_position.x );
	measurement_( 1, 0 ) = static_cast<float>( measured_position.y );

	// The "correct" phase that is going to use the predicted value and our measurement
	cv::Mat estimated = kalman_filter_.correct( measurement_ );
	cv::Point2f statePt( estimated.at< float >( 0 ), estimated.at< float >( 1 ) );

	return cv::Point2i{ cl::math::RoundToLong( statePt.x ), cl::math::RoundToLong( statePt.y ) };
}

//-------------------------------------------------------------------------------------------------
//
bool
ObjectTracker::has_polygon()
{
	return has_polygon_;
}

//-------------------------------------------------------------------------------------------------
//
bool
ObjectTracker::is_in_ellipse( const cv::RotatedRect& ellipse )
{
	float theta = ellipse.angle;
	cv::Point center = ellipse.center;
	cv::Point p = get_predicted_position();
	double u = cos( theta ) * (p.x - center.x) + sin( theta ) * (p.y - center.y);
	double v = -sin( theta ) * (p.x - center.x) + cos( theta ) * (p.y - center.y);
	double dist = sqrt( pow( u / ellipse.size.width, 2 ) + pow( v / ellipse.size.height, 2 ) );

	if( dist <= 1 )
	{
		return true;
	}

	return false;
}

//-------------------------------------------------------------------------------------------------
//
bool
ObjectTracker::is_in_bouding_rect( const cv::Rect& rect )
{
	return rect.contains( get_predicted_position() );
}

//-------------------------------------------------------------------------------------------------
//
bool
ObjectTracker::is_in_polygon( const PolygonPtr& poly )
{
	double distance = cv::pointPolygonTest( poly->contour(), get_predicted_position(), false );

	if( distance >= 0 )
	{
		return true;
	}
	return false;
}

//-------------------------------------------------------------------------------------------------
//
void
ObjectTracker::draw_ellipse( cv::Mat& to, const cv::Scalar& color )
{
	ellipse( to, ellipse_, color, 1, CV_AA );
}

//-------------------------------------------------------------------------------------------------
//
void
ObjectTracker::update_mean_ellipse( const cv::RotatedRect& ellipse, cv::Point2i p )
{
	cv::Size2f mean_size{ };
	float mean_angle{ };

	ellipses_.push_back( ellipse );

	for( auto e : ellipses_ )
	{
		mean_size.width += e.size.width;
		mean_size.height += e.size.height;
		mean_angle += e.angle;
	}

	mean_size.width /= static_cast<float>(ellipses_.size());
	mean_size.height /= static_cast<float>(ellipses_.size());
	mean_angle /= static_cast<float>(ellipses_.size());

	ellipse_ = cv::RotatedRect( p, mean_size, mean_angle );
}
