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

#ifndef OBJECTTRACKER_H
#define OBJECTTRACKER_H

//=================================================================================================
// I N C L U D E   F I L E S

#include <opencv2/core/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <memory>
#include <CLCircularBuffer.h>
#include <CLUtility.h>
#include "CLMath.h"


//=================================================================================================
// F O R W A R D   D E C L A R A T I O N S

//=================================================================================================
// C O N S T A N T S

//=================================================================================================
// C L A S S E S

typedef std::vector< cv::Point > Contour;

struct Polygon
{
//--Methods----------------------------------------------------------------------------------------
public:
	Polygon() = default;

	~Polygon() = default;

	void initialize( const Contour& contour_passed )
	{
		Contour contour_tmp( contour_passed.size() );

		cv::approxPolyDP( contour_passed, contour_tmp, 1, true );
		contour_ = contour_tmp;
		area_ = cv::contourArea( contour_ );
		bound_rect_ = boundingRect( cv::Mat( contour_ ) );

		cv::Point2f tmp;
		minEnclosingCircle( static_cast<cv::Mat>(contour_), tmp, radius_ );

		center_ = cv::Point2i( cl::math::RoundToLong( tmp.x ),
		                       cl::math::RoundToLong( tmp.y ) );

		cv::Moments m = moments( contour_, false );

		int32_t x = cl::math::RoundToLong( m.m10 / m.m00 );
		int32_t y = cl::math::RoundToLong( m.m01 / m.m00 );

		moment_ = cv::Point2i( x, y );
	}

	void draw_contour( cv::Mat& to, const cv::Scalar& color )
	{
		std::vector< Contour > tmp;
		tmp.push_back( contour_ );
		drawContours( to, tmp, -1, color, 1, CV_AA );
	}

	void draw_min_ellipse( cv::Mat& to, const cv::Scalar& color )
	{
		cl::ignore( to );
		cl::ignore( color );
		min_ellipse_ = cv::fitEllipse( cv::Mat( contour_ ) );
		//ellipse( to, min_ellipse_, color, 1, CV_AA );
	}

	bool is_touching_edge()
	{
		for( const auto& point : contour_ )
		{
			if( point.y == 1 || point.y == 374 - 2 )
			{
				return true;
			}
		}
		return false;
	}

	const Contour& contour()
	{
		return contour_;
	}

	const cv::Point2i& moment()
	{
		return moment_;
	}

	const cv::RotatedRect& min_ellipse()
	{
		return min_ellipse_;
	}

	double area()
	{ return area_; }

	const cv::Rect& bound_rect()
	{ return bound_rect_; }

	const cv::Point2i& center()
	{ return center_; }

//--Data members-----------------------------------------------------------------------------------
	Contour contour_;
private:
	double area_;
	cv::Rect bound_rect_;
	cv::Point2i center_;
	float radius_;
	cv::Point2i moment_;
	cv::RotatedRect min_ellipse_;
};

typedef std::shared_ptr< Polygon > PolygonPtr;

class ObjectTracker
{
//--Methods----------------------------------------------------------------------------------------
public:
	ObjectTracker();

	~ObjectTracker();

	void initialize( const cv::Point2i& measured_position, uint32_t index );

	void compute_prediction();

	cv::Point2i get_predicted_position();

	cv::Point2i get_estimated_position( const cv::Point2i& measured_position );

	bool has_polygon();

	bool is_in_ellipse( const cv::RotatedRect& ellipse );

	bool is_in_bouding_rect( const cv::Rect& rect );

	bool is_in_polygon( const PolygonPtr& poly );

	void update_mean_ellipse( const cv::RotatedRect& ellipse, cv::Point2i p );

	void draw_ellipse( cv::Mat& to, const cv::Scalar& color );

//--Data members-----------------------------------------------------------------------------------
private:
	cv::Point2f predicted_position_;
	cv::KalmanFilter kalman_filter_;
	cv::Mat_< float > measurement_;

	cl::CircularBuffer< cv::RotatedRect > ellipses_;

public:
	bool has_polygon_;
	uint32_t lost_count_;
	uint32_t index_;

	cv::RotatedRect ellipse_;
};

typedef std::shared_ptr< ObjectTracker > ObjectTrackerPtr;

//=================================================================================================
// I N L I N E   F U N C T I O N S   C O D E   S E C T I O N

#endif  // OBJECTTRACKER_H
