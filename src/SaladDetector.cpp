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

#include "SaladDetector.h"

//==================================================================================================
// C O N S T A N T S   &   L O C A L   V A R I A B L E S

//==================================================================================================
// G L O B A L S

//==================================================================================================
// C O N S T R U C T O R (S) / D E S T R U C T O R   C O D E   S E C T I O N

//--------------------------------------------------------------------------------------------------
//
SaladDetector::SaladDetector()
	: salad_count_{ }
{ }

//--------------------------------------------------------------------------------------------------
//
SaladDetector::~SaladDetector()
{ }

//=================================================================================================
// M E T H O D S   C O D E   S E C T I O N

//--------------------------------------------------------------------------------------------------
//
void
SaladDetector::set_src_img( const cv::Mat& src_img )
{
	src_img_ = src_img;
}

//--------------------------------------------------------------------------------------------------
//
void
SaladDetector::set_erode( int32_t erode_size )
{
	element_erode_ = cv::getStructuringElement( cv::MORPH_ELLIPSE,
	                                            cv::Size( 2 * erode_size + 1, 2 * erode_size + 1 ),
	                                            cv::Point( erode_size, erode_size ) );
}

//--------------------------------------------------------------------------------------------------
//
void
SaladDetector::set_dilate( int32_t dilate_size )
{
	element_dilate_ = cv::getStructuringElement( cv::MORPH_ELLIPSE,
	                                             cv::Size( 2 * dilate_size + 1, 2 * dilate_size + 1 ),
		                                         cv::Point( dilate_size, dilate_size ) );
}

//--------------------------------------------------------------------------------------------------
//
void
SaladDetector::init_blob_detector( bool area, bool circularity, bool color, bool convexity, bool inertia )
{
	simple_blob_params_.filterByArea        = area;
	simple_blob_params_.filterByCircularity = circularity;
	simple_blob_params_.filterByColor       = color;
	simple_blob_params_.filterByConvexity   = convexity;
	simple_blob_params_.filterByInertia     = inertia;
}
//--------------------------------------------------------------------------------------------------
//
void
SaladDetector::set_blob_detector_threshold( float_t min, float_t max )
{
	simple_blob_params_.minThreshold = min;
	simple_blob_params_.maxThreshold = max;
}

//--------------------------------------------------------------------------------------------------
//
void
SaladDetector::set_blob_detector_area_range( float_t min, float_t max )
{
	simple_blob_params_.minArea = min;
	simple_blob_params_.maxArea = max;
}

//--------------------------------------------------------------------------------------------------
//
void
SaladDetector::set_blob_detector_circularity_range( float_t min, float_t max )
{
	simple_blob_params_.minCircularity = min;
	simple_blob_params_.maxCircularity = max;
}

//--------------------------------------------------------------------------------------------------
//
void
SaladDetector::set_blob_detector_color( uchar color )
{
	simple_blob_params_.blobColor = color;
}

//--------------------------------------------------------------------------------------------------
//
void
SaladDetector::set_blob_detector_convexity_range( float_t min, float_t max )
{
	simple_blob_params_.minConvexity = min;
	simple_blob_params_.maxConvexity = max;
}

//--------------------------------------------------------------------------------------------------
//
void
SaladDetector::set_blob_detector_inertia_range( float_t min, float_t max )
{
	simple_blob_params_.minInertiaRatio = min;
	simple_blob_params_.maxInertiaRatio = max;
}

//--------------------------------------------------------------------------------------------------
//
void
SaladDetector::compute()
{
	cv::Mat tmp;

	// convert src to lab, invert, threshold, erode and dilate A channel
	cv::cvtColor( src_img_, tmp, CV_BGR2Lab );
	cv::split( tmp, split_lab_img_ );
	cv::bitwise_not( split_lab_img_[1], tmp );
	cv::threshold( tmp, thres_img_, 0, 255 / 2, cv::THRESH_BINARY | cv::THRESH_OTSU );
	cv::erode( thres_img_, eroded_img_, element_erode_ );
	cv::dilate( eroded_img_, dilated_img_, element_dilate_ );

	hulls_img_ = src_img_.clone();

	// find contours of thresholded, eroded and dilated image
	std::vector< std::vector< cv::Point > > contours;
	tmp = dilated_img_.clone();
	cv::findContours( tmp, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE );

	// get hulls from contours
	std::vector< std::vector < int32_t > > hulls_int( contours.size() );
	std::vector< std::vector< cv::Point > > hulls( contours.size() );
	for( size_t i = 0; i < contours.size(); ++i )
	{
		cv::convexHull( cv::Mat( contours[i] ), hulls_int[i], false );

		for( size_t k = 0; k < hulls_int[i].size(); ++k )
		{
			int32_t ind{ hulls_int[i][k] };
			hulls[i].emplace_back( cv::Point( contours[i][ind].x, contours[i][ind].y ) );
		}
	}

	// remove hulls too small
	auto hulls_it = hulls.begin();
	while( hulls_it != hulls.end() )
	{
		float_t area{ 0.0f };
		for( size_t i = 0; i < (*hulls_it).size(); ++i )
		{
			size_t next_i = ( i + 1 ) % (*hulls_it).size();
			area += static_cast< float_t >( (*hulls_it)[i].x * (*hulls_it)[next_i].y );
			area -= static_cast< float_t >( (*hulls_it)[next_i].x * (*hulls_it)[i].y );
		}
		area /= 2.0f;

		// area can be negative
		if( area < 0 )
		{
			area *= -1;
		}
		//cl::print_line( "Area: ", area );

		if( area < 200 )
		{
			hulls_it = hulls.erase( hulls_it );
		}
		else
		{
			++hulls_it;
		}
	}

	// remove hulls that touches edge
	hulls_it = hulls.begin();
	while( hulls_it != hulls.end() )
	{
		bool on_edge{ false };
		for( size_t i = 0; i < (*hulls_it).size(); ++i )
		{
			if( 750 > (*hulls_it)[i].x && (*hulls_it)[i].x < 2 )
			{
				on_edge = true;
			}
		}
		if( on_edge )
		{
			hulls_it = hulls.erase( hulls_it );
		}
		else
		{
			++hulls_it;
		}
	}

	// simple blob detect
	cv::SimpleBlobDetector simple_blob_detector( simple_blob_params_ );
	std::vector< cv::KeyPoint > key_points;
	simple_blob_detector.detect( hulls_img_, key_points );

	// erase keypoints which are not in a hull
	auto key_points_it = key_points.begin();
	while( key_points_it != key_points.end() )
	{
		bool is_in_hull{ false };
		for( size_t i = 0; i < hulls.size(); ++i )
		{
			if( cv::pointPolygonTest( hulls[i], (*key_points_it).pt, false ) > 0.0 )
			{
				is_in_hull = true;
			}
		}
		if( !is_in_hull )
		{
			key_points_it = key_points.erase( key_points_it );
		}
		else
		{
			++key_points_it;
		}
	}

	// erase hulls without keypoints
	hulls_it = hulls.begin();
	while( hulls_it != hulls.end() )
	{
		bool has_key_point{ false };
		for( size_t i = 0; i < key_points.size(); ++i )
		{
			if( cv::pointPolygonTest( (*hulls_it), key_points[i].pt, false > 0.0 ) )
			{
				has_key_point = true;
			}
		}
		if( !has_key_point )
		{
			hulls_it = hulls.erase( hulls_it );
		}
		else
		{
			++hulls_it;
		}
	}


	//// draw hull contours
	//if( hulls.size() )
	//{
	//	hulls_img_ = src_img_.clone();
	//	cv::drawContours( hulls_img_, hulls, -1, cv::Scalar( 0, 255, 255 ) );
	//}

	// draw keypoints with hulls
	if( key_points.size() )
	{
		cv::drawKeypoints( src_img_, key_points, key_points_img_, cv::Scalar( 255, 255, 255 ),
		                   cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
	}

	// get center of polys
	std::vector< cv::Point > poly_moments;
	for( const auto& hull : hulls )
	{
		cv::Moments m = cv::moments( hull, false );
		cv::Point pt( cl::math::RoundToLong( m.m10 / m.m00 ),
		              cl::math::RoundToLong( m.m01 / m.m00 ) );
		poly_moments.push_back( pt );
	}

	// merge close points
	auto poly_moments_it = poly_moments.begin();
	while( poly_moments_it != poly_moments.end() )
	{
		auto poly_moments_next_it = poly_moments_it + 1;
		while( poly_moments_next_it != poly_moments.end() )
		{
			float_t dist{ sqrtf( powf( (*poly_moments_it).x - (*poly_moments_next_it).x, 2.0f ) +
			                     powf( (*poly_moments_it).y - (*poly_moments_next_it).y, 2.0f ) ) };
			if( dist < 30.0f )
			{
				(*poly_moments_it) = cv::Point( ( (*poly_moments_it).x +
				                                  (*poly_moments_next_it).x ) / 2,
				                                ( (*poly_moments_it).y +
				                                  (*poly_moments_next_it).y ) / 2 );
				poly_moments_next_it = poly_moments.erase( poly_moments_next_it );
			}
			else
			{
				++poly_moments_next_it;
			}
		}
		++poly_moments_it;
	}
	cl::print_line( "pas boucle 5" );


	// grid for checking
	std::vector< cv::Point > frg; // four rows grid
	std::vector< cv::Point > trg; // three rows grid
	for( size_t i = 0; i < 4; ++i )
	{
		for( size_t k = 0; k < 3; ++k )
		{
			int32_t x = static_cast< int32_t >(k) * 180;
			int32_t y = static_cast< int32_t >(i) * 140;
			frg.push_back( cv::Point( x, y ) );
			if( i < 3 )
			{
				trg.push_back( cv::Point( x, y ) );
			}
		}
	}

	// find the furthest point of grid of 4 rows
	int32_t frg_max_x { };
	int32_t frg_max_y { };
	for( const auto& pt : frg )
	{
		if( pt.x > frg_max_x )
		{
			frg_max_x = pt.x;
		}
		if( pt.y > frg_max_y )
		{
			frg_max_y = pt.y;
		}
	}

	// check 4 rows grid matching everywhere on image and select best match
	int32_t frg_best_offset_x { };
	int32_t frg_best_offset_y { };
	float_t frg_best_ratio { };
	float_t frg_best_avg_dist { 100.0f };
	float_t frg_min_ratio { static_cast< float_t >( frg.size()  - 2 ) /
	                                   static_cast< float_t >( frg.size() ) };

	int32_t frg_offset_y { };
	while( ( frg_offset_y + frg_max_y) < 480 )
	{
		int32_t frg_offset_x { };
		while( ( frg_offset_x + frg_max_x) < 752 )
		{
			int32_t frg_good_points_count { };
			float_t frg_avg_dist { };

			for( const auto& grid_pt : frg )
			{
				for( const auto& moment_pt : poly_moments )
				{
					if( grid_pt.x + frg_offset_x + 50 > moment_pt.x &&
						moment_pt.x > grid_pt.x + frg_offset_x - 50 &&
						grid_pt.y + frg_offset_y + 50 > moment_pt.y &&
						moment_pt.y > grid_pt.y + frg_offset_y - 50 )
					{
						++frg_good_points_count;
						frg_avg_dist += sqrtf( powf( grid_pt.x + frg_offset_x - moment_pt.x, 2.0f ) +
						                       powf( grid_pt.y + frg_offset_y - moment_pt.y, 2.0f ) );
					}
				}
			}

			// calculate best average dist between grid points and poly centers to get best grid
			frg_avg_dist /= static_cast< float_t >( frg_good_points_count );
			float_t frg_ratio { static_cast< float_t >( frg_good_points_count ) /
			                    static_cast< float_t >( frg.size() ) };

			// update datas of best grid found
			if( frg_best_ratio <= frg_ratio &&
				frg_ratio >= frg_min_ratio &&
				frg_avg_dist <= frg_best_avg_dist )
			{
				frg_best_ratio = frg_ratio;
				frg_best_offset_x = frg_offset_x;
				frg_best_offset_y = frg_offset_y;
				frg_best_avg_dist = frg_avg_dist;
			}
			frg_offset_x += 5;
		}
		frg_offset_y += 5;
	}

	// find the furthest point of grid of 3 rows
	int32_t trg_max_x { };
	int32_t trg_max_y { };
	for( const auto& pt : trg )
	{
		if( pt.x > trg_max_x )
		{
			trg_max_x = pt.x;
		}
		if( pt.y > trg_max_y )
		{
			trg_max_y = pt.y;
		}
	}

	// check 3 rows grid matching everywhere on image and select best match
	int32_t trg_best_offset_x { };
	int32_t trg_best_offset_y { };
	float_t trg_best_ratio { };
	float_t trg_best_avg_dist { 100.0f };
	float_t trg_min_ratio { static_cast< float_t >( trg.size() - 2 ) /
	                        static_cast< float_t >( trg.size()) };

	int32_t trg_offset_y { };
	while((trg_offset_y + trg_max_y) < 480 )
	{
		int32_t trg_offset_x { };
		while((trg_offset_x + trg_max_x) < 752 )
		{
			int32_t trg_good_points_count { };
			float_t trg_avg_dist { };

			for( const auto& grid_pt : trg )
			{
				for( const auto& moment_pt : poly_moments )
				{
					if( grid_pt.x + trg_offset_x + 50 > moment_pt.x &&
					    moment_pt.x > grid_pt.x + trg_offset_x - 50 &&
					    grid_pt.y + trg_offset_y + 50 > moment_pt.y &&
					    moment_pt.y > grid_pt.y + trg_offset_y - 50 )
					{
						++trg_good_points_count;
						trg_avg_dist += sqrtf( powf( grid_pt.x + trg_offset_x - moment_pt.x, 2.0f ) +
						                       powf( grid_pt.y + trg_offset_y - moment_pt.y, 2.0f ) );
					}
				}
			}

			// calculate best average dist between grid points and poly centers to get best grid
			trg_avg_dist /= static_cast< float_t >( trg_good_points_count );
			float_t trg_ratio { static_cast< float_t >( trg_good_points_count ) /
			                    static_cast< float_t >( trg.size() ) };

			// update datas of best grid found
			if( trg_best_ratio <= trg_ratio &&
			    trg_ratio >= trg_min_ratio &&
			    trg_avg_dist <= trg_best_avg_dist )
			{
				trg_best_ratio = trg_ratio;
				trg_best_offset_x = trg_offset_x;
				trg_best_offset_y = trg_offset_y;
				trg_best_avg_dist = trg_avg_dist;
			}
			trg_offset_x += 5;
		}
		trg_offset_y += 5;
	}

	// if ratio is good enough
	if( frg_best_ratio > frg_min_ratio )
	{
		// erase hulls without grid
		hulls_it = hulls.begin();
		while( hulls_it != hulls.end() )
		{
			cv::Moments m = cv::moments( (*hulls_it), false );
			cv::Point pt( cl::math::RoundToLong( m.m10 / m.m00 ),
			              cl::math::RoundToLong( m.m01 / m.m00 ) );

			bool is_in_best_grid{ };
			for( const auto& grid_pt : frg )
			{
				if( grid_pt.x + frg_best_offset_x + 50 > pt.x &&
					pt.x > grid_pt.x + frg_best_offset_x - 50 &&
					grid_pt.y + frg_best_offset_y + 50 > pt.y &&
					pt.y > grid_pt.y + frg_best_offset_y - 50 )
				{
					is_in_best_grid = true;
				}
			}

			if( !is_in_best_grid )
			{
				hulls_it = hulls.erase( hulls_it );
			}
			else
			{
				++hulls_it;
			}
		}
	}
	else if( trg_best_ratio > trg_min_ratio )
	{
		// erase hulls without grid
		hulls_it = hulls.begin();
		while( hulls_it != hulls.end())
		{
			cv::Moments m = cv::moments((*hulls_it), false );
			cv::Point pt( cl::math::RoundToLong( m.m10 / m.m00 ),
			              cl::math::RoundToLong( m.m01 / m.m00 ));

			bool is_in_best_grid { };
			for( const auto& grid_pt : trg )
			{
				if( grid_pt.x + trg_best_offset_x + 50 > pt.x &&
				    pt.x > grid_pt.x + trg_best_offset_x - 50 &&
				    grid_pt.y + trg_best_offset_y + 50 > pt.y &&
				    pt.y > grid_pt.y + trg_best_offset_y - 50 )
				{
					is_in_best_grid = true;
				}
			}

			if( !is_in_best_grid )
			{
				hulls_it = hulls.erase( hulls_it );
			}
			else
			{
				++hulls_it;
			}
		}
	}
	else
	{
		hulls.resize( 0 );
	}

	// draw hull contours
	if( hulls.size())
	{
		hulls_img_ = src_img_.clone();
		cv::drawContours( hulls_img_, hulls, -1, cv::Scalar( 0, 255, 255 ));
	}

	hulls_poly_img_ = hulls_img_.clone();
	// kalman
	// save hulls which don't touch the top or bottom edge
	std::vector< PolygonPtr > hulls_poly;
	for( const auto& hull : hulls )
	{
		PolygonPtr poly_ptr( std::make_shared< Polygon >() );
		poly_ptr->initialize( hull );
		if( !poly_ptr->is_touching_edge() )
		{
			poly_ptr->draw_min_ellipse( hulls_poly_img_, cv::Scalar( 0, 255, 0 ) );
			poly_ptr->draw_contour( hulls_poly_img_, cv::Scalar( 0, 255, 0 ) );
			hulls_poly.push_back( poly_ptr );
		}
	}

	// erase lost objects
	tracked_objects_.erase( std::remove_if( tracked_objects_.begin(),
	                                        tracked_objects_.end(),
	                                        []( ObjectTrackerPtr obj )
	                                        {
		                                        return obj->lost_count_ > 15;
	                                        }),
	                      tracked_objects_.end() );

	// update prediction for all tracked objects
	for( const auto& tracked : tracked_objects_ )
	{
		tracked->compute_prediction();
		tracked->has_polygon_ = false;
	}

	// track with polys

	// used to extend bounding rects in case the tracked object
	// in case it goes out of bounds (usually in the begining of tracking)
	int32_t extension{ 30 };
	cv::Point estimated;

	auto hulls_poly_it = hulls_poly.begin();
	while( hulls_poly_it != hulls_poly.end() )
	{
		bool already_tracked{ };

		// extend bounding rect of poly
		cv::Rect bounding_rect_extended = (*hulls_poly_it)->bound_rect();
		if( bounding_rect_extended.x > extension )
			bounding_rect_extended.x -= extension;
		if( bounding_rect_extended.y > extension )
			bounding_rect_extended.y -= extension;
		bounding_rect_extended.height += extension * 2;
		bounding_rect_extended.width += extension * 2;

		for( auto& tracked : tracked_objects_ )
		{
			// Correct and draw prediction
			if( tracked->is_in_bouding_rect( bounding_rect_extended ))
			{
				estimated = tracked->get_estimated_position((*hulls_poly_it)->moment());
				tracked->update_mean_ellipse((*hulls_poly_it)->min_ellipse(), estimated );
				tracked->has_polygon_ = true;
				tracked->lost_count_ = 0;
				already_tracked = true;
				break;
			}
		}

		// if poly is not already tracked, track it
		if( !already_tracked )
		{
			ObjectTrackerPtr newt = std::make_shared< ObjectTracker >();
			tracked_objects_.push_back( newt );

			newt->initialize((*hulls_poly_it)->moment(), ++salad_count_ );

			newt->has_polygon_ = true;
			estimated = newt->get_estimated_position((*hulls_poly_it)->moment());

			newt->update_mean_ellipse((*hulls_poly_it)->min_ellipse(), estimated );
			//newt->draw_ellipse( sourceL, scalar_green );
			//
			//cv::circle( sourceL, estimated, 2, scalar_green, -1, CV_AA );
			//putText( sourceL, std::to_string( newt->index_ ), estimated,
			//         CV_FONT_NORMAL, 0.5, scalar_white, 1, CV_AA );
		}

		hulls_poly_it = hulls_poly.erase( hulls_poly_it );
	}

	output_img_ = src_img_.clone();

	for( const auto& tracked : tracked_objects_ )
	{
		if( tracked->has_polygon())
		{
			estimated = tracked->get_position();
			tracked->draw_ellipse( output_img_, cv::Scalar( 0, 255, 0 ) );

			cv::circle( output_img_, estimated, 2, cv::Scalar( 0, 255, 0 ), -1, CV_AA );
			putText( output_img_, std::to_string( tracked->index_ ), estimated,
			         CV_FONT_NORMAL, 0.5, cv::Scalar( 255, 255, 255 ), 1, CV_AA );
		}
		else
		{
			++tracked->lost_count_;
		}
	}
}

//--------------------------------------------------------------------------------------------------
//
const std::vector< cv::Mat >
SaladDetector::get_split_src_img()
{
	return split_lab_img_;
}

//--------------------------------------------------------------------------------------------------
//
const cv::Mat
SaladDetector::get_thres_img()
{
	return thres_img_;
}

//--------------------------------------------------------------------------------------------------
//
const cv::Mat
SaladDetector::get_eroded_img()
{
	return eroded_img_;
}

//--------------------------------------------------------------------------------------------------
//
const cv::Mat
SaladDetector::get_dilated_img()
{
	return dilated_img_;
}

//--------------------------------------------------------------------------------------------------
//
const cv::Mat
SaladDetector::get_hulls_img()
{
	return hulls_img_;
}

//--------------------------------------------------------------------------------------------------
//
const cv::Mat
SaladDetector::get_key_points_img()
{
	return key_points_img_;
}

//--------------------------------------------------------------------------------------------------
//
const cv::Mat
SaladDetector::get_hulls_poly_img()
{
	return hulls_poly_img_;
}

//--------------------------------------------------------------------------------------------------
//
const cv::Mat
SaladDetector::get_output_img()
{
	return output_img_;
}

