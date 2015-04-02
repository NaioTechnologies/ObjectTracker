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
#include "Histogram.h"
#include "EntryPoint.h"
#include "BuildVersion.h"

#include "HTUtility.h"
#include "CLFileSystem.h"

#include <opencv2/highgui/highgui.hpp>
#include <ObjectTracker.h>


//==================================================================================================
// C O N S T A N T S   &   L O C A L   V A R I A B L E S

//==================================================================================================
// G L O B A L S

//==================================================================================================
// C O N S T R U C T O R (S) / D E S T R U C T O R   C O D E   S E C T I O N

EntryPoint::EntryPoint()
	: parser_{ }
	, handler_{ }
	, pressed_{ false }
	, ul_{ }
	, dr_{ }
{
	// Register program info
	std::string appName{ PROGRAM_NAME };
	appName.append( "-" );
	appName.append( PROGRAM_MAJOR_VERSION );
	appName.append( "." );
	appName.append( PROGRAM_MINOR_VERSION );

	// Set logger output name for the program
	HTLogger::SetExecModuleName( appName );

	//auto f = std::bind( &EntryPoint::handle_parameters, this, std::placeholders::_1,
	//                    std::placeholders::_2 );
	//handler_.AddParamHandler( "-c", f );
	//parser_.add_switch( "-c", "Calibrate stereo bench" );
}

//--------------------------------------------------------------------------------------------------
//
EntryPoint::~EntryPoint()
{ }


//=================================================================================================
// M E T H O D S   C O D E   S E C T I O N

//-------------------------------------------------------------------------------------------------
//
void
EntryPoint::print_header() const
{
	using namespace cl;
	print_line( "=============================================================================" );
	print_line();
	print_line( "  ", PROGRAM_NAME, " version ", PROGRAM_MAJOR_VERSION, ".", PROGRAM_MINOR_VERSION,
	            ".", PROGRAM_PATCH_VERSION, "  Copyright(c) 2013  ", PROGRAM_OWNER );
	print_line( "  ", PROGRAM_DESCRIPTION );
	print_line( "" );
	print_line( "  This program is free software: you can redistribute it and/or modify" );
	print_line( "  it under the terms of the GNU General Public License as published by" );
	print_line( "  the Free Software Foundation, either version 3 of the License, or" );
	print_line( "  (at your option) any later version." );
	print_line( "" );
	print_line( "  This program is distributed in the hope that it will be useful," );
	print_line( "  but WITHOUT ANY WARRANTY; without even the implied warranty of" );
	print_line( "  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the" );
	print_line( "  GNU General Public License for more details." );
	print_line();
	print_line( "=============================================================================" );
}

//-------------------------------------------------------------------------------------------------
//
bool
EntryPoint::handle_parameters( const std::string& paramName, const std::string& paramValue )
{
	cl::ignore( paramName );
	cl::ignore( paramValue );
	//if( paramName.compare( "-c" ) )
	//{
	//	if( paramValue.compare( "true" ) )
	//	{
	//		return true;
	//	}
	//
	//	cl::print_line( "Unexpected parameter" );
	//}
	//
	return false;
}

void
EntryPoint::call_back_func( int event, int x, int y, int flags, void* userdata )
{
	EntryPoint* core = reinterpret_cast<EntryPoint*>( userdata );

	if( event == cv::EVENT_LBUTTONDOWN )
	{
		core->pressed_ = true;
		core->ul_ = cv::Point2i( x, y );
	}
	else if( event == cv::EVENT_LBUTTONUP )
	{
		core->pressed_ = false;
		core->dr_ = cv::Point2i( x, y );
	}
	else if( event == cv::EVENT_MOUSEMOVE )
	{
		core->dr_ = cv::Point2i( x, y );
	}
}

void FindBlobs( const cv::Mat& binary, std::vector< std::vector< cv::Point2i>>& blobs )
{
	blobs.clear();

	// Fill the label_image with the blobs
	// 0  - background
	// 1  - unlabelled foreground
	// 2+ - labelled foreground
	cv::Mat label_image;
	binary.convertTo( label_image, CV_32SC1 );

	int label_count = 2; // starts at 2 because 0,1 are used already

	for( int y = 0; y < label_image.rows; y++ )
	{
		int* row = (int*) label_image.ptr( y );
		for( int x = 0; x < label_image.cols; x++ )
		{
			if( row[x] != 0 )
			{
				continue;
			}

			cv::Rect rect;
			cv::floodFill( label_image, cv::Point( x, y ), label_count, &rect, 0, 0, 4 );

			std::vector< cv::Point2i > blob;

			for( int i = rect.y; i < (rect.y + rect.height); i++ )
			{
				int* row2 = (int*) label_image.ptr( i );
				for( int j = rect.x; j < (rect.x + rect.width); j++ )
				{
					if( row2[j] != label_count )
					{
						continue;
					}

					blob.push_back( cv::Point2i( j, i ) );
				}
			}

			blobs.push_back( blob );

			label_count++;
		}
	}
}

//-------------------------------------------------------------------------------------------------
//
int32_t
EntryPoint::run( int32_t argc, const char** argv )
{
	int32_t res{ };
	print_header();

	if( parser_.validate_cmd_line( argc, argv, &handler_ ) )
	{
		ht::log_info( "Entering main loop" );

		cv::Size size( 752, 480 );

		uint8_t key_pressed{ };
		bool need_hist_reset{ }, has_valid_hist{ };
		cv::MatND ab_hist;
		cv::Rect rect( cv::Rect( 752 / 2 - 20, 480 / 2 - 20, 40, 40 ) );
		cv::Mat hist_roi( rect.height - 2, rect.width - 2, CV_8UC3 );

		ColorHistogram colorHist;
		ObjectFinder finder;

		int32_t iLastX = -1;
		int32_t iLastY = -1;

		int32_t erosion_size = 5;
		cv::Mat element = cv::getStructuringElement( cv::MORPH_ELLIPSE,
		                                             cv::Size( 2 * erosion_size + 1,
		                                                       2 * erosion_size + 1 ),
		                                             cv::Point( erosion_size, erosion_size ) );

		std::string folderName{ "/media/jean/Data/Capture/Salads/Wed_Dec_17_11_58_42_2014_+0100" };

		if( !cl::filesystem::folder_exists( folderName ) )
		{
			throw cl::Exception( "Folder does not exist", CL_ORIGIN );
		}

		//Create a window
		cv::namedWindow( "final_output" );

		std::vector< ObjectTrackerPtr > tracked_objects;

		uint64_t image_nbr{ 10 };
		uint32_t index{ };
		uint32_t obj_size_threshold{ 1000 };

		/// Draw contours
		cv::Scalar scalar_green( 0, 255, 0 );
		cv::Scalar scalar_red( 0, 0, 255 );
		cv::Scalar scalar_blue( 255, 0, 0 );
		cv::Scalar scalar_yellow( 0, 255, 255 );
		cv::Scalar scalar_white( 255, 255, 255 );

		while( key_pressed != 27 )
		{
			cv::Mat sourceL = cv::Mat( size, CV_8UC3 );

			std::string filePathL{ folderName };
			filePathL.append( "/" );
			filePathL.append( std::to_string( image_nbr ) );
			filePathL.append( "_l.tiff" );

			sourceL = cv::imread( filePathL );

			//std::string filePathR{ folderName };
			//filePathR.append( "/" );
			//filePathR.append( std::to_string( image_nbr ) );
			//filePathR.append( "_r.tiff" );
			//
			//mat_r = cv::imread( filePathR );

			//cv::Mat3b rectified_l = cv::Mat3b::zeros( size_rectified );
			//cv::Mat3b rectified_r = cv::Mat3b::zeros( size_rectified );
			//
			//rectifier.generate_rectified_mat( mat_l, mat_r, rectified_l, rectified_r );

			cv::Mat src_cpy = sourceL.clone();
			cv::Mat intermediate = sourceL.clone();
			//cv::GaussianBlur( src_cpy, src_cpy, cv::Size(5, 5), 5 );

			//cv::Mat sharpenedL;
			//cv::GaussianBlur( sourceL, sharpenedL, cv::Size(0, 0), 3 );
			//cv::addWeighted( sourceL, 1.5, sharpenedL, -0.5, 0, sharpenedL );

			//set the callback function for any mouse event
			cv::setMouseCallback( "final_output", &EntryPoint::call_back_func, this );

			//cv::Mat3b wb = cv::Mat3b::zeros( size_rectified );
			cv::Mat3b wb;

			if( !has_valid_hist && pressed_ )
			{
				rect = cv::Rect( ul_, dr_ );
				cv::rectangle( src_cpy, rect, cv::Scalar( 0, 255, 0 ) );
			}

			if( need_hist_reset )
			{
				cv::rectangle( src_cpy, rect, cv::Scalar( 0, 255, 0 ) );

				cl::print_line( "Reseting histogram..." );

				cv::Mat hsv;
				//cv::cvtColor( mat_l, hsv, CV_BGR2HSV );
				//hist_roi = hsv( cv::Rect(rect.x+1, rect.y+1, rect.width-2, rect.height-2) );
				hist_roi =
					src_cpy( cv::Rect( rect.x + 1, rect.y + 1, rect.width - 2, rect.height - 2 ) );

				ab_hist = colorHist.get_ab_histogram( hist_roi );
				//ab_hist = colorHist.getHueHistogram( hist_roi );
				//ab_hist = colorHist.getHistogram( hist_roi );

				finder.set_histogram( ab_hist );
				finder.set_threshold( 0.05 );

				need_hist_reset = false;
				has_valid_hist = true;
			}

			if( has_valid_hist )
			{
				++image_nbr;
				cl::print_line( "\nLoop number ", image_nbr );

				// Convert to Lab space
				cv::Mat lab2, hsv;
				cv::cvtColor( src_cpy, lab2, CV_BGR2Lab );
				//cv::cvtColor( mat_l, hsv, CV_BGR2HSV );

				// Get back-projection of ab histogram
				int32_t ch[2] = { 1, 2 };
				cv::Mat result_ab = finder.find( lab2, 0, 255, ch, 2 );

				//int32_t ch[3] = {0, 1, 2};
				//cv::Mat result_ab = finder.find( hsv, 0, 255, ch, 3 );

				// Erode and dilate result
				//cv::dilate( result_ab, result_ab, element_1, cv::Point( -1, -1 ), 1 );
				//cv::erode( result_ab, result_ab, element_1, cv::Point( -1, -1 ), 2 );

				///// Apply Laplace function
				//Laplacian( result_ab, result_ab, CV_16S, 3, 1, 0, BORDER_DEFAULT );
				//convertScaleAbs( result_ab, result_ab );

				/// Detect edges using canny
				//Canny( canny, canny, 20, 500, 3 );

				//cv::Mat test = result_ab.clone();
				//std::vector<std::vector<cv::Point2i>> blobs;
				//
				//FindBlobs( test, blobs );
				//
				//cl::PrintLine( blobs.size() );
				//
				//// Randomy color the blobs
				//for( size_t i = 0; i < blobs.size(); i++ )
				//{
				//	unsigned char r = 255 * (rand() / (1.0 + RAND_MAX));
				//	unsigned char g = 255 * (rand() / (1.0 + RAND_MAX));
				//	unsigned char b = 255 * (rand() / (1.0 + RAND_MAX));
				//
				//	for( size_t j = 0; j < blobs[i].size(); j++ )
				//	{
				//		int x = blobs[i][j].x;
				//		int y = blobs[i][j].y;
				//
				//		src_cpy.at<cv::Vec3b>( y, x )[0] = b;
				//		src_cpy.at<cv::Vec3b>( y, x )[1] = g;
				//		src_cpy.at<cv::Vec3b>( y, x )[2] = r;
				//	}
				//}

				std::vector< Contour > raw_contours;
				std::vector< cv::Vec4i > hierarchy;

				// Find the raw contours in the image
				cv::Mat canny = result_ab.clone();
				findContours( canny, raw_contours, hierarchy, CV_RETR_EXTERNAL,
				              CV_CHAIN_APPROX_SIMPLE );

				//std::vector<Contour> raw_contours_resampled( raw_contours.size() );
				//for( size_t i = 0; i < raw_contours.size(); ++i )
				//{
				//	ResampleCurve( raw_contours[i], raw_contours_resampled[i], 200, false );
				//}

				//std::vector< std::vector<int> > hull( raw_contours.size() );
				//std::vector< std::vector<cv::Vec4i> > convDef( raw_contours.size() );
				//std::vector< std::vector< cv::Point> > hull_points( raw_contours.size() );
				//std::vector< std::vector< cv::Point> > defect_points( raw_contours.size() );
				//
				//std::vector<PolygonPtr> valid_poly;
				//std::vector<PolygonPtr> possible_valid_poly;
				//std::vector<PolygonPtr> invalid_poly;
				//
				//// Create all polygons and separate the valid polygons
				//for( size_t i = 0; i < raw_contours.size(); ++i )
				//{
				//	PolygonPtr poly( std::make_shared<Polygon>() );
				//	poly->initialize( raw_contours.at( i ) );
				//
				//	// There is an object to track in the image
				//	if( poly->area() > obj_size_threshold )
				//	{
				//		//convexHull( raw_contours[i], hull[i], false );
				//		//convexityDefects( raw_contours[i], hull[i], convDef[i] );
				//		//
				//		//for( int k = 0; k < hull[i].size(); k++ )
				//		//{
				//		//	int ind = hull[i][k];
				//		//	hull_points[i].push_back( raw_contours[i][ind] );
				//		//}
				//		//
				//		//for( int k = 0; k < convDef[i].size(); k++ )
				//		//{
				//		//	if( convDef[i][k][3] > 20 * 256 )
				//		//	{
				//		//		int ind_0 = convDef[i][k][0];
				//		//		int ind_1 = convDef[i][k][1];
				//		//		int ind_2 = convDef[i][k][2];
				//		//		defect_points[i].push_back( raw_contours[i][ind_2] );
				//		//		cv::circle( rectified_l, raw_contours[i][ind_0], 5, Scalar( 0, 255, 0 ), -1 );
				//		//		cv::circle( rectified_l, raw_contours[i][ind_1], 5, Scalar( 0, 255, 0 ), -1 );
				//		//		cv::circle( rectified_l, raw_contours[i][ind_2], 5, Scalar( 0, 0, 255 ), -1 );
				//		//		cv::line( rectified_l, raw_contours[i][ind_2], raw_contours[i][ind_0],
				//		//		          Scalar( 0, 0, 255 ), 1 );
				//		//		cv::line( rectified_l, raw_contours[i][ind_2], raw_contours[i][ind_1],
				//		//		          Scalar( 0, 0, 255 ), 1 );
				//		//	}
				//		//}
				//
				//		if( !poly->is_touching_edge() )
				//		{
				//			poly->draw_min_ellipse( sourceL, scalar_green );
				//			poly->draw_contour( intermediate, scalar_green );
				//			valid_poly.push_back( poly );
				//		}
				//		else
				//		{
				//			poly->draw_contour( intermediate, scalar_yellow );
				//			possible_valid_poly.push_back( poly );
				//		}
				//	}
				//	else
				//	{
				//		poly->draw_contour( intermediate, scalar_red );
				//		invalid_poly.push_back( poly );
				//	}
				//}
				//
				////auto iter = valid_poly.begin();
				////while( iter != valid_poly.end() )
				////{
				////	// Check if polygon could be splitted into two polys
				////
				////
				////	poly->area() > obj_size_threshold
				////
				////	++iter;
				////}
				//
				//tracked_objects.erase( std::remove_if( tracked_objects.begin(), tracked_objects.end(),
				//                                       []( ObjectTrackerPtr obj )
				//                                       {
				//                                           return obj->lost_count_ > 200;
				//                                       } ),
				//                       tracked_objects.end() );
				//
				//// Update prediction for all tracked objects
				//for( const auto& tracked : tracked_objects )
				//{
				//	tracked->compute_prediction();
				//	tracked->has_polygon_ = false;
				//}
				//
				//auto it = valid_poly.begin();
				//while( it != valid_poly.end() )
				//{
				//	bool already_tracked {};
				//	for( auto& tracked : tracked_objects )
				//	{
				//		if( tracked->is_in_bouding_rect( (*it)->bound_rect() ) )
				//		{
				//			cv::Point2i estimated = tracked->get_estimated_position( (*it)->moment() );
				//
				//			tracked->update_mean_ellipse( (*it)->min_ellipse(), estimated );
				//			tracked->draw_ellipse( sourceL, scalar_green );
				//
				//			cv::circle( sourceL, estimated, 2, scalar_green, -1, CV_AA );
				//			putText( sourceL, std::to_string( tracked->index_ ), estimated,
				//			         CV_FONT_NORMAL, 0.5, scalar_white, 1, CV_AA );
				//
				//			tracked->has_polygon_ = true;
				//
				//			already_tracked = true;
				//			break;
				//		}
				//	}
				//
				//	if( !already_tracked )
				//	{
				//		ObjectTrackerPtr newt = std::make_shared<ObjectTracker>();
				//		tracked_objects.push_back( newt );
				//
				//		newt->initialize( (*it)->moment(), ++index );
				//
				//
				//		newt->has_polygon_ = true;
				//		cv::Point2i estimated = newt->get_estimated_position( (*it)->moment() );
				//
				//		newt->update_mean_ellipse( (*it)->min_ellipse(), estimated );
				//		newt->draw_ellipse( sourceL, scalar_green );
				//
				//		cv::circle( sourceL, estimated, 2, scalar_green, -1, CV_AA );
				//		putText( sourceL, std::to_string( newt->index_ ), estimated,
				//		         CV_FONT_NORMAL, 0.5, scalar_white, 1, CV_AA );
				//	}
				//
				//	it = valid_poly.erase( it );
				//}
				//
				//for( const auto& tracked : tracked_objects )
				//{
				//	if( !tracked->has_polygon() )
				//	{
				//		++tracked->lost_count_;
				//	}
				//}
				//cv::imshow( "Result ab", result_ab );
				cv::cvtColor( result_ab, wb, CV_GRAY2BGR );
			}

			//cv::imshow( "left", rectified_l );

			cv::Mat combined( size.height * 2, size.width * 2, CV_8UC3 );

			cv::Mat source( combined, cv::Rect( 0, 0, size.width, size.height ) );
			src_cpy.copyTo( source );

			cv::Mat wb2( combined, cv::Rect( size.width, 0, size.width, size.height ) );
			wb.copyTo( wb2 );

			cv::Mat intermediate2( combined, cv::Rect( 0, size.height, size.width, size.height ) );
			intermediate.copyTo( intermediate2 );

			cv::Mat final( combined, cv::Rect( size.width, size.height, size.width, size.height ) );
			sourceL.copyTo( final );

			//cv::Mat1b src_r( sharpened_combined, cv::Rect( size.width, 0, size.width, size.height ) );
			//sharpened_r.copyTo( mat_r );

			cv::imshow( "final_output", combined );

			key_pressed = static_cast<uint8_t>( cv::waitKey( 10 ) );
			if( key_pressed == static_cast<uint8_t>('r') )
			{
				need_hist_reset = true;
			}
		}

		ht::log_info( "Exiting main loop" );
	}

	return res;
}


//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//--------------------------------------- Main Function -------------------------------------------
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

int32_t
main( int32_t argc, const char** argv )
{
	int32_t ret{ };

	try
	{
		EntryPoint ep;
		ret = ep.run( argc, argv );
	}
	catch( const cl::SystemError& e )
	{
		ht::log_fatal( std::string( "System Error: " ).append( e.what() ).append( e.where() ) );
		throw;
	}
	catch( const cl::BaseException& e )
	{
		ht::log_fatal( std::string( "CLBaseException: " ).append( e.what() ) );
		throw;
	}
	catch( const std::exception& e )
	{
		ht::log_fatal( std::string( "std::exception caught: " ).append( e.what() ) );
		throw;
	}
	catch( ... )
	{
		ht::log_fatal( "Caught an exception of an undetermined type" );
		throw;
	}

	return ret;
}
