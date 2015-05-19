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
#include "SaladDetector.h"

#include "IO/IOMVBlueFoxManager.h"
#include "IO/IOSessionLoader.hpp"

#include "HTUtility.h"
#include "CLFileSystem.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <ObjectTracker.h>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/objdetect/objdetect.hpp>

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
	//handler_.AddParamHandler( "-s", f );
	//parser_.add_argument( "-s", "object size" );
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

		int32_t iLastX{ -1 };
		int32_t iLastY{ -1 };

		int32_t erosion_size{ 3 };
		int32_t dilate_size{ 3 };
		int32_t area_erase{ 200 };
		int32_t defect_depth{ 20 };
		int32_t defect_lenght{ 100 };

		float_t fBlobMinThres{ 0.0 };
		float_t fBlobMaxThres{ 1.0 };
		int32_t iBlobMinThres{ 0 };
		int32_t iBlobMaxThres{ 255 };

		float_t fBlobMinCirc{ 0.0 };
		float_t fBlobMaxCirc{ 1.0 };
		int32_t iBlobMinCirc{ 50-40/*+0*/ };
		int32_t iBlobMaxCirc{ 100 };

		float_t fBlobMinConvex{ 0.0 };
		float_t fBlobMaxConvex{ 1.0 };
		int32_t iBlobMinConvex{ 80-80/*+10*/ };
		int32_t iBlobMaxConvex{ 100 };

		float_t fBlobMinInertia{ 0.0 };
		float_t fBlobMaxInertia{ 1.0 };
		int32_t iBlobMinInertia{ 40-30/*+0*/ };
		int32_t iBlobMaxInertia{ 100 };

		float_t fBlobMinArea{ 0.0 };
		float_t fBlobMaxArea{ 1000.0 };
		int32_t iBlobMinArea{ 20-18/*+2*/ };
		int32_t iBlobMaxArea{ 100 };

		int32_t minDist{ 1 };

		//cv::Mat element = cv::getStructuringElement( cv::MORPH_ELLIPSE,
		//                                             cv::Size( 2 * erosion_size + 1,
		//                                                       2 * erosion_size + 1 ),
		//                                             cv::Point( erosion_size, erosion_size ) );

		//std::string folderName{ "/home/robin/Programming/ObjectTracker/img/big/" };
		//std::string folderName{ "/home/robin/Programming/ObjectTracker/img/field_bright1/" };
		//std::string folderName{ "/home/robin/Programming/ObjectTracker/img/field_bright2/" };
		std::string folderName{ "/home/robin/Programming/ObjectTracker/img/small/" };
		//std::string folderName{ "/home/robin/Documents/Mon_May__4_16_02_28_2015_+0200/" };
		//std::string folderName{ "/home/robin/Documents/Mon_May__4_16_05_38_2015_+0200/" };
		//std::string folderName{ "/home/robin/Documents/Mon_May__4_16_21_10_2015_+0200/" };
		//std::string folderName{ "/home/robin/Documents/Mon_May__4_16_22_09_2015_+0200/" };

		if( !cl::filesystem::folder_exists( folderName ) )
		{
			throw cl::Exception( "Folder does not exist", CL_ORIGIN );
		}

		//Create a window
		//cv::namedWindow( "final_output" );

		//std::vector< ObjectTrackerPtr > tracked_objects;
		std::vector< ObjectTrackerPtr > trackedObjects;

		uint64_t image_nbr{ 10 };
		uint32_t index{ };
		uint32_t obj_size_threshold{ 1000 };

		/// Draw contours
		cv::Scalar scalar_green( 0, 255, 0 );
		cv::Scalar scalar_red( 0, 0, 255 );
		cv::Scalar scalar_blue( 255, 0, 0 );
		cv::Scalar scalar_yellow( 0, 255, 255 );
		cv::Scalar scalar_white( 255, 255, 255 );

		//io::MVBlueFoxManager cameraManager;
		//io::MVBlueFoxPtr camera = cameraManager.get_device( 0 );
		//
		//camera->open( ht::ColorSpace::Rgb );
		//camera->set_auto_exposure( true );

		io::SessionLoader session_loader( folderName, 5 );
		session_loader.begin_transfer();
		ht::BitmapUniquePtr bm = session_loader.get_next_entry();
		cv::Mat3b src_img = cv::Mat3b( bm->height(), bm->width());
		SaladDetector salad_detector;

		while( key_pressed != 27 )
		{
			src_img.data = bm->data();
			salad_detector.set_src_img( src_img );
			cv::cvtColor( src_img, src_img, CV_RGB2BGR );
			salad_detector.set_erode( 3 );
			salad_detector.set_dilate( 3 );
			salad_detector.init_blob_detector( true, true, false, true, false );
			salad_detector.set_blob_detector_threshold( 0.0f, 255.0f );
			salad_detector.set_blob_detector_area_range( 200.0f, 1000000.0f );
			salad_detector.set_blob_detector_circularity_range( 0.1f, 1.0f );
			salad_detector.set_blob_detector_convexity_range( 0.0f, 1.0f );
			salad_detector.set_blob_detector_inertia_range( 0.1f, 1.0f );
			salad_detector.compute();


			cv::Mat hulls_img = salad_detector.get_hulls_img();
			cv::Mat key_points_img = salad_detector.get_key_points_img();
			cv::Mat hulls_poly_img = salad_detector.get_hulls_poly_img();
			cv::Mat out_img = salad_detector.get_output_img();
			cv::imshow( "src", src_img );
			cv::imshow( "hulls", hulls_img );
			//cv::imshow( "key_points", key_points_img );
			//cv::imshow( "hulls_poly", hulls_poly_img );
			cv::imshow( "out", out_img );
			bm = session_loader.get_next_entry();

			/*////io::FramePtr frame = camera->wait_for_image( 250 );

			cv::Mat sourceL = cv::Mat( size, CV_8UC3 );

			////cl::BufferUniquePtr buffer = frame->extract_data();
			////sourceL.data = buffer->data();
			////cv::Mat cam = cv::Mat( size, CV_8UC3 );
			////cam.data = buffer->data();
			////cv::imshow("Camera", cam);

			std::string filePathL{ folderName };
			filePathL.append( std::to_string( image_nbr ) );
			filePathL.append( "_l.tiff" );
			////filePathL.append( "_r.tiff" );
			////filePathL.append( ".tiff" );

			sourceL = cv::imread( filePathL );

			//// reduce brightness
			////cv::Mat sourceL_cpy = sourceL.clone();
			////sourceL_cpy -= cv::Scalar::all(75);
			////cv::addWeighted(sourceL, 0.5, sourceL_cpy, 0.5, 0, sourceL);
			//
			////std::string filePathR{ folderName };
			////filePathR.append( "/" );
			////filePathR.append( std::to_string( image_nbr ) );
			////filePathR.append( "_r.tiff" );
			////
			////mat_r = cv::imread( filePathR );
			//
			////cv::Mat3b rectified_l = cv::Mat3b::zeros( size_rectified );
			////cv::Mat3b rectified_r = cv::Mat3b::zeros( size_rectified );
			////
			////rectifier.generate_rectified_mat( mat_l, mat_r, rectified_l, rectified_r );

			cv::Mat src_cpy = sourceL.clone();
			cv::Mat intermediate = sourceL.clone();

			////cv::GaussianBlur( src_cpy, src_cpy, cv::Size(5, 5), 5 );
			//
			////cv::Mat sharpenedL;
			////cv::GaussianBlur( sourceL, sharpenedL, cv::Size(0, 0), 3 );
			////cv::addWeighted( sourceL, 1.5, sharpenedL, -0.5, 0, sharpenedL );

			////set the callback function for any mouse event
			//cv::setMouseCallback( "final_output", &EntryPoint::call_back_func, this );

			////cv::Mat3b wb = cv::Mat3b::zeros( size_rectified );
			////cv::Mat3b wb;
			cv::Mat wb = sourceL.clone();
			//std::vector< cv::Mat > vectLab2;

			if( !has_valid_hist && pressed_ )
			{
				rect = cv::Rect( ul_, dr_ );
				cv::rectangle( src_cpy, rect, cv::Scalar( 0, 255, 0 ) );
			}

			if( need_hist_reset )
			{
				cv::rectangle( src_cpy, rect, cv::Scalar( 0, 255, 0 ) );

				cl::print_line( "Reseting histogram..." );

			//	cv::Mat hsv;
			//	//cv::cvtColor( mat_l, hsv, CV_BGR2HSV );
			//	//hist_roi = hsv( cv::Rect(rect.x+1, rect.y+1, rect.width-2, rect.height-2) );
			//	hist_roi =
			//		src_cpy( cv::Rect( rect.x + 1, rect.y + 1, rect.width - 2, rect.height - 2 ) );
			//
			//	ab_hist = colorHist.get_ab_histogram( hist_roi );
			//	//ab_hist = colorHist.getHueHistogram( hist_roi );
			//	//ab_hist = colorHist.getHistogram( hist_roi );
			//
			//	finder.set_histogram( ab_hist );
			//	finder.set_threshold( 0.05 );

				need_hist_reset = false;
				has_valid_hist = true;
			}

			if( has_valid_hist )
			{
				++image_nbr;
				if(image_nbr > 700) image_nbr = 10;
				cl::print_line( "\nLoop number ", image_nbr );

				// Convert to Lab space
				//cv::Mat lab2; //, hsv;

				// threshold on lab
				//std::vector<cv::Mat> channels;
				//cv::Mat src_eq;
				//cv::cvtColor(src_cpy, src_eq, CV_BGR2YCrCb);
				//cv::split(src_eq, channels);
				//cv::equalizeHist(channels[0], channels[0]);
				//cv::merge(channels, src_eq);
				//cv::cvtColor(src_eq, src_cpy, CV_YCrCb2BGR);
				//cv::createTrackbar( "Gaussian kernels", "Dilated", &gaussian_kernels, 31 );
				//cv::GaussianBlur( src_cpy, src_cpy,
				//                  cv::Size( gaussian_kernels * 2 + 1, gaussian_kernels * 2 + 1 ), 0,
				//                  0 );
				//cv::blur(src_cpy, src_cpy, cv::Size( gaussian_kernels * 2 + 1, gaussian_kernels * 2 + 1 ));
				//cv::medianBlur(src_cpy, src_cpy, gaussian_kernels*2 + 1);

				// Extract A from LAB
				cv::Mat imgLab;
				cv::cvtColor( src_cpy, imgLab, CV_BGR2Lab );
				std::vector< cv::Mat > splitLab;
				cv::split( imgLab, splitLab );
				cv::Mat imgL = splitLab[0].clone();
				cv::Mat imgA = splitLab[1].clone();
				cv::Mat imgB = splitLab[2].clone();

				// invert colors
				cv::Mat imgAInv;
				imgAInv = imgA.clone();
				cv::bitwise_not( imgA, imgAInv );

				//cv::Mat imgAInvEq;
				//cv::equalizeHist( imgAInv, imgAInvEq );
				//cv::equalizeHist( imgA, imgA );

				// add threshold
				cv::Mat imgThres;
				cv::threshold( imgAInv, imgThres, 0, 255 / 2, cv::THRESH_BINARY | cv::THRESH_OTSU );
				//cv::threshold( imgA, imgThres, 0, 255 / 2, cv::THRESH_BINARY | cv::THRESH_OTSU );

				// erode and dilate type and size
				//cv::createTrackbar( "Erode size", "Final", &erosion_size, 21 );
				//cv::createTrackbar( "Dilate size", "Final", &dilate_size, 21 );
				//cv::createTrackbar( "Area erase", "Final", &area_erase, 1000 );
				//cv::createTrackbar( "Defect depth", "Final", &defect_depth, 50);
				//cv::createTrackbar( "Defect length", "Final", &defect_lenght, 200);
				cv::Mat element_erode = cv::getStructuringElement( cv::MORPH_ELLIPSE,
				                                                   cv::Size( 2 * erosion_size + 1,
				                                                             2 * erosion_size + 1 ),
				                                                   cv::Point( erosion_size,
				                                                              erosion_size ) );
				cv::Mat element_dilate = cv::getStructuringElement( cv::MORPH_ELLIPSE,
				                                                    cv::Size( 2 * dilate_size + 1,
				                                                              2 * dilate_size + 1 ),
				                                                    cv::Point( dilate_size,
				                                                               dilate_size ) );
				// eroding and dilating
				cv::Mat imgEroded;
				cv::erode( imgThres, imgEroded, element_erode );
				cv::Mat imgDilated;
				cv::dilate( imgEroded, imgDilated, element_dilate );
				//cv::erode( imgDilated, imgEroded, element_erode );
				//cv::dilate( imgEroded, imgDilated, element_dilate );

				//cv::cvtColor( mat_l, hsv, CV_BGR2HSV );

				// Get back-projection of ab histogram
				//int32_t ch[2] = { 1, 2 };
				//cv::Mat result_ab = finder.find( lab2, 0, 255, ch, 2 );

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

				// Find the raw contours in the image
				//cv::Mat canny = result_ab.clone();

				// to find the contours
				std::vector< Contour > raw_contours;
				std::vector< cv::Vec4i > hierarchy;
				cv::Mat imgDilatedCopy = imgDilated.clone();
				cv::findContours( imgDilatedCopy, raw_contours, hierarchy, CV_RETR_EXTERNAL,
				              CV_CHAIN_APPROX_SIMPLE );
				//cv::drawContours(src_cpy, raw_contours, -1, scalar_green);

				// draw convex hull
				std::vector< std::vector< int32_t > > hullI( raw_contours.size() );

				// to use to correct hulls
				std::vector< std::vector< cv::Point > > hullPoints( raw_contours.size() );
				//std::vector< std::vector< cv::Vec4i > > defects( raw_contours.size());
				//std::vector< std::vector< cv::Point > > defectStartPoints( raw_contours.size());
				//std::vector< std::vector< cv::Point > > defectEndPoints( raw_contours.size());
				//std::vector< std::vector< cv::Point > > defectMaxPoints( raw_contours.size());

				// for each contour found
				for( size_t i = 0; i < raw_contours.size(); i++ )
				{
					// finding convex hull of contour i
					cv::convexHull( cv::Mat( raw_contours[i] ), hullI[i], false );

					// getting coordinates of the hull points (probably better than doing another cv::convexHull)
					for( size_t k = 0; k < hullI[i].size(); k++ )
					{
						int32_t ind{ hullI[i][k] };
						hullPoints[i].emplace_back( cv::Point(raw_contours[i][ind].x,
						                                        raw_contours[i][ind].y) );
					}

					// calculating hull area
					double_t area{ 0.0 };
					for( size_t k = 0; k < hullPoints[i].size(); k++)
					{
						size_t kNext = (k + 1) % hullPoints[i].size();
						area += static_cast< double_t >(hullPoints[i][k].x * hullPoints[i][kNext].y );
						area -= static_cast< double_t >(hullPoints[i][kNext].x * hullPoints[i][k].y );
					}
					area /= 2.0;
					// area can be negative
					if( area < 0 ) area *= -1;
					// erasing the hull for which area is too small (small weeds, ...)
					if( area < area_erase )
					{
						hullPoints[i].clear();
					}

					// getting hull defects points and drawing them
					//if( hullI[i].size() > 3 )
					//{
					//	cv::convexityDefects( raw_contours[i], hullI[i], defects[i] );
					//	for( int k = 0; k < defects[i].size(); k++ )
					//	{
					//		double startMaxLenght = 0;
					//		double maxEndLenght = 0;f
					//
					//		if( defects[i][k][3] > defect_depth * 256 ) // filters defects by depth
					//		{
					//			int ind_0 =
					//				defects[i][k][0]; // index to point of the contour where defect begins
					//			int ind_1 =
					//				defects[i][k][1]; // index to point of the contour where defect ends
					//			int ind_2 =
					//				defects[i][k][2]; // index to point the farthest point from the hull within the defect
					//			//defectStartPoints[i]
					//			//	.push_back( raw_contours[i][ind_0] ); // to use to correct hulls
					//			//defectEndPoints[i]
					//			//	.push_back( raw_contours[i][ind_1] ); // to use to correct hulls
					//			//defectMaxPoints[i]
					//			//	.push_back( raw_contours[i][ind_2] ); // to use to correct hulls
					//			//cv::circle( src_cpy, raw_contours[i][ind_0], 5,
					//			//            scalar_green, -1 ); // highlight the defect begin point
					//			//cv::circle( src_cpy, raw_contours[i][ind_1], 5,
					//			//            scalar_blue, -1 ); // highlight the defect end point
					//			//cv::circle( src_cpy, raw_contours[i][ind_2], 5,
					//			//            scalar_red,
					//			//            -1 ); // highlight the farthest point from the hull within the defect
					//			//cv::line( src_cpy, raw_contours[i][ind_2],
					//			//          raw_contours[i][ind_0],
					//			//          scalar_red,
					//			//          1 ); // draw a line between the farthest point and the defect begin point
					//			//cv::line( src_cpy, raw_contours[i][ind_2],
					//			//          raw_contours[i][ind_1],
					//			//          scalar_red,
					//			//          1 ); // draw a line between the farthest point and the defect end point
					//
					//			//  correcting bad hulls
					//			//  Idea 1:
					//			//      if too much lenght between start point or end point and max point:
					//			//          replace start/end point by max point
					//
					//			/*//*
					//			// start point to max point lenght
					//			startMaxLenght = sqrt(
					//				pow( raw_contours[i][ind_0].x - raw_contours[i][ind_2].x,
					//				     2.0 ) +
					//				pow( raw_contours[i][ind_0].y - raw_contours[i][ind_2].y,
					//				     2.0 ));
					//			// max point to end point lenght
					//			maxEndLenght = sqrt(
					//				pow( raw_contours[i][ind_2].x - raw_contours[i][ind_1].x,
					//				     2.0 ) +
					//				pow( raw_contours[i][ind_2].y - raw_contours[i][ind_1].y,
					//				     2.0 ));
					//			//cl::print_line(
					//			//	"-----------------------------------------------------" );
					//			//cl::print_line( "(", i, "-", k, ")" );
					//			//cl::print_line( "StartMax: ", startMaxLenght );
					//			//cl::print_line( "MaxEnd:   ", maxEndLenght );
					//			//cl::print_line( "Start:    ", raw_contours[i][ind_0] );
					//			//cl::print_line( "End:      ", raw_contours[i][ind_1] );
					//			//cl::print_line( "Max:      ", raw_contours[i][ind_2] );
					//
					//			if( startMaxLenght >= maxEndLenght )
					//			{
					//				if( startMaxLenght > defect_lenght )
					//				{
					//					int l;
					//					// search start point in hull
					//					for( l = 0; l < hullPoints[i].size(); l++ )
					//						if( hullPoints[i][l] == raw_contours[i][ind_0] )
					//						{
					//							// replace start point by max point
					//							hullPoints[i][l] = raw_contours[i][ind_2];
					//							break;
					//						}
					//				}
					//			}
					//			else
					//			{
					//				if( maxEndLenght > defect_lenght )
					//				{
					//					int l;
					//					// search end point in hull
					//					for( l = 0; l < hullPoints[i].size(); l++)
					//						if( hullPoints[i][l] == raw_contours[i][ind_1] )
					//						{
					//							// replace end point by max point
					//							hullPoints[i][l] = raw_contours[i][ind_2];
					//							break;*//*10*//*
					//						}
					//				}
					//			}
					//
					//			/*//*//*
					//
					//			//  Idea 2:
					//			//      number of points of hull proportional to area of hull
					//
					//			//  Idea 3:
					//			//      if 2 max points close and lenght to start/end point too long:
					//			//          merge points
					//		}
					//	}
					//}
				}

				// display hull points
				//for( int i = 0; i < hullPoints.size(); i++ )
				//{
				//	for( int k = 0; k < hullPoints[i].size(); k++ )
				//	{
				//		cv::circle( src_cpy, hullPoints[i][k], 2, scalar_red, -1 );
				//	}
				//}

				// erase empty hulls (crashes pointPolygonTest)
				auto hullPointsIt = hullPoints.begin();
				while( hullPointsIt != hullPoints.end())
				{
					if((*hullPointsIt).size() == 0 )
					{
						hullPointsIt = hullPoints.erase( hullPointsIt );
					}
					else
					{
						++hullPointsIt;
					}
				}

				hullPointsIt = hullPoints.begin();
				while( hullPointsIt != hullPoints.end() )
				{
					bool onEdge{ false };
					for( int32_t i = 0; i < (*hullPointsIt).size(); ++i )
					{
						if( 750 > (*hullPointsIt)[i].x && (*hullPointsIt)[i].x < 2)
						{
							onEdge = true;
						}
					}
					if( onEdge )
					{
						hullPointsIt = hullPoints.erase( hullPointsIt );
					}
					else
					{
						++hullPointsIt;
					}
				}

				// draw hull contours on first image
				// absolutely in green, yellow or white for the blob detection
				if( hullPoints.size() )
					cv::drawContours( src_cpy, hullPoints, -1, scalar_yellow );
				//
				// SimpleBlobDetector
				// simple blob detect
				// Setup parameters
				cv::SimpleBlobDetector::Params simpleBlobParams;

				cv::createTrackbar( "Blob min thres", "Final", &iBlobMinThres, 255 );
				cv::createTrackbar( "Blob max thres", "Final", &iBlobMaxThres, 255 );
				cv::createTrackbar( "Blob min circ (/100)", "Final", &iBlobMinCirc, 100 );
				cv::createTrackbar( "Blob max circ (/100)", "Final", &iBlobMaxCirc, 100 );
				cv::createTrackbar( "Blob min convex (/100)", "Final", &iBlobMinConvex, 100 );
				cv::createTrackbar( "Blob max convex (/100)", "Final", &iBlobMaxConvex, 100 );
				cv::createTrackbar( "Blob min inertia (/100)", "Final", &iBlobMinInertia, 100 );
				cv::createTrackbar( "Blob max inertia (/100)", "Final", &iBlobMaxInertia, 100 );
				cv::createTrackbar( "Blob min area", "Final", &iBlobMinArea, 100 );
				cv::createTrackbar( "Blob max area", "Final", &iBlobMaxArea, 100 );
				fBlobMinThres = static_cast< float_t >(iBlobMinThres);
				fBlobMaxThres = static_cast< float_t >(iBlobMaxThres);
				fBlobMinCirc = static_cast< float_t >(iBlobMinCirc) / 100.0f;
				fBlobMaxCirc = static_cast< float_t >(iBlobMaxCirc) / 100.0f;
				fBlobMinConvex = static_cast< float_t >(iBlobMinConvex) / 100.0f;
				fBlobMaxConvex = static_cast< float_t >(iBlobMaxConvex) / 100.0f;
				fBlobMinInertia = static_cast< float_t >(iBlobMinInertia) / 100.0f;
				fBlobMaxInertia = static_cast< float_t >(iBlobMaxInertia) / 100.0f;
				fBlobMinArea = static_cast< float_t >(iBlobMinArea);
				fBlobMaxArea = static_cast< float_t >(iBlobMaxArea);

				simpleBlobParams.minThreshold = fBlobMinThres;
				simpleBlobParams.maxThreshold = fBlobMaxThres;

				simpleBlobParams.filterByArea = true;
				simpleBlobParams.minArea = fBlobMinArea * 100;
				simpleBlobParams.maxArea = fBlobMaxArea * 10000;

				simpleBlobParams.filterByCircularity = true;
				simpleBlobParams.minCircularity = fBlobMinCirc;
				simpleBlobParams.maxCircularity = fBlobMaxCirc;

				simpleBlobParams.filterByColor = false;

				simpleBlobParams.filterByConvexity = true;
				simpleBlobParams.minConvexity = fBlobMinConvex;
				simpleBlobParams.maxConvexity = fBlobMaxConvex;

				simpleBlobParams.filterByInertia = true;
				simpleBlobParams.minInertiaRatio = fBlobMinInertia;
				simpleBlobParams.maxInertiaRatio = fBlobMaxInertia;

				// detect
				cv::SimpleBlobDetector detector( simpleBlobParams );

				std::vector< cv::KeyPoint > keyPoints;
				detector.detect( src_cpy, keyPoints );

				// erase keypoint which are not in a hull
				auto keyPointsIt = keyPoints.begin();
				bool isInHull{ false };
				while( keyPointsIt != keyPoints.end() )
				{
					isInHull = false;
					for( size_t i = 0; i < hullPoints.size(); ++i )
					{
						if( cv::pointPolygonTest(hullPoints[i], (*keyPointsIt).pt, false ) > 0.0 )
						{
							isInHull = true;
						}
					}
					if( !isInHull )
					{
						keyPointsIt = keyPoints.erase( keyPointsIt );
					}
					else
					{
						++keyPointsIt;
					}
				}

				// draw keypoints on intermediate image
				if( keyPoints.size() )
					cv::drawKeypoints( src_cpy, keyPoints, intermediate,
					                   scalar_white,
					                   cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

				// erase hulls without key point
				hullPointsIt = hullPoints.begin();
				bool hasKeyPoint{ false };
				while( hullPointsIt != hullPoints.end())
				{
					hasKeyPoint = false;
					for( size_t i = 0; i < keyPoints.size(); i++ )
					{
						if ( cv::pointPolygonTest( (*hullPointsIt), keyPoints[i].pt, false ) > 0.0 )
						{
							hasKeyPoint = true;
						}
					}
					if( !hasKeyPoint )
					{
						hullPointsIt = hullPoints.erase( hullPointsIt );
					}
					else
					{
						++hullPointsIt;
					}
				}
				//
				// grid for checking
				std::vector< cv::Point2i > grid4Rows;
				std::vector< cv::Point2i > grid3Rows;
				for( size_t i = 0; i < 4; ++i )
				{
					for( size_t j = 0; j < 3; ++j )
					{
						int32_t x = static_cast<int32_t>(j) * 180;
						int32_t y = static_cast<int32_t>(i) * 140;
						grid4Rows.push_back( cv::Point( x, y ));
						if( i < 3 )
						{
							grid3Rows.push_back( cv::Point( x, y ));
						}
					}
				}


				// detect centers of polys
				std::vector< cv::Point2i > polyMoments;
				for( const auto& hull : hullPoints )
				{
					cv::Moments m = moments( hull, false );
					cv::Point2i pt( cl::math::RoundToLong( m.m10 / m.m00 ),
					                cl::math::RoundToLong( m.m01 / m.m00 ));
					polyMoments.push_back( pt );
				}

				// merge close points
				auto polyMomentsIt = polyMoments.begin();
				while( polyMomentsIt != polyMoments.end())
				{
					auto checkPolyMomentsIt = polyMomentsIt + 1;
					while( checkPolyMomentsIt != polyMoments.end())
					{
						float_t dist
							{ sqrtf( powf((*polyMomentsIt).x - (*checkPolyMomentsIt).x, 2.0f ) +
							         powf((*polyMomentsIt).y - (*checkPolyMomentsIt).y, 2.0f )) };
						if( dist < 30.0f )
						{
							(*polyMomentsIt) = cv::Point(((*polyMomentsIt).x +
							                              (*checkPolyMomentsIt).x) / 2,
							                             ((*polyMomentsIt).y +
							                              (*checkPolyMomentsIt).y) / 2 );
							checkPolyMomentsIt = polyMoments.erase(checkPolyMomentsIt);
						}
						else
						{
							++checkPolyMomentsIt;
						}
					}
					cv::circle(src_cpy, (*polyMomentsIt), 2, scalar_red, 2);
					++polyMomentsIt;
				}

				//std::vector< std::vector< cv::Point2i > > listAlignedPointsX;
				//std::vector< std::vector< cv::Point2i > > listAlignedPointsY;
				//
				//for( const auto& moment1 : polyMoments )
				//{
				//	bool hasAlignedPointsX{ false };
				//	bool hasAlignedPointsY{ false };
				//	std::vector< cv::Point2i > alignedPointsX;
				//	std::vector< cv::Point2i > alignedPointsY;
				//
				//	for( const auto& moment2 : polyMoments )
				//	{
				//		auto itX = std::find( alignedPointsX.begin(), alignedPointsX.end(),
				//		                      moment2 );
				//		if( itX == alignedPointsX.end())
				//		{
				//			if( moment1 != moment2 )
				//			{
				//				if( moment2.x > moment1.x - 30 && moment2.x < moment1.x + 30 )
				//				{
				//					alignedPointsX.push_back( moment2 );
				//					hasAlignedPointsX = true;
				//				}
				//			}
				//		}
				//
				//		auto itY = std::find( alignedPointsY.begin(), alignedPointsY.end(),
				//		                      moment2 );
				//		if( itY == alignedPointsY.end())
				//		{
				//			if( moment1 != moment2 )
				//			{
				//				if( moment2.y > moment1.y - 30 && moment2.y < moment1.y + 30 )
				//				{
				//					alignedPointsY.push_back( moment2 );
				//					hasAlignedPointsY = true;
				//				}
				//			}
				//		}
				//	}
				//
				//	if( hasAlignedPointsX )
				//	{
				//		alignedPointsX.push_back( moment1 );
				//	}
				//	if( hasAlignedPointsY )
				//	{
				//		alignedPointsY.push_back( moment1 );
				//	}
				//
				//	bool isInListX{ false };
				//	for( const auto& pointsX : alignedPointsX )
				//	{
				//		auto listItX = listAlignedPointsX.begin();
				//		while( listItX != listAlignedPointsX.end() )
				//		{
				//			auto it = std::find( (*listItX).begin(), (*listItX).end(), pointsX );
				//			if( it != (*listItX).end() )
				//			{
				//				isInListX = true;
				//			}
				//			++listItX;
				//		}
				//	}
				//
				//	bool isInListY { false };
				//	for( const auto& pointsY : alignedPointsY )
				//	{
				//		auto listItY = listAlignedPointsY.begin();
				//		while( listItY != listAlignedPointsY.end())
				//		{
				//			auto it = std::find((*listItY).begin(), (*listItY).end(), pointsY );
				//			if( it != (*listItY).end())
				//			{
				//				isInListY = true;
				//			}
				//			++listItY;
				//		}
				//	}
				//
				//	if( alignedPointsX.size() != 0 && !isInListX )
				//	{
				//		listAlignedPointsX.push_back( alignedPointsX );
				//	}
				//	if( alignedPointsY.size() != 0 && !isInListY )
				//	{
				//		listAlignedPointsY.push_back( alignedPointsY );
				//	}
				//}

				// TODO : grid check with translation of model

				//==================================================================================
				// 4 ROWS

				// find farthest point of grid of 4 rows
				int32_t gridMaxX4Rows { };
				int32_t gridMaxY4Rows { };
				for( size_t i = 0; i < grid4Rows.size(); ++i )
				{
					if( grid4Rows[i].x > gridMaxX4Rows )
					{
						gridMaxX4Rows = grid4Rows[i].x;
					}
					if( grid4Rows[i].y > gridMaxY4Rows )
					{
						gridMaxY4Rows = grid4Rows[i].y;
					}
				}

				int32_t bestOffsetX4Rows { };
				int32_t bestOffsetY4Rows { };
				float_t bestRatio4Rows { };
				float_t bestAvgPointDist4Rows { 100.0f };
				float_t minRatio4Rows { static_cast< float_t >( grid4Rows.size() - 2 ) /
				                        static_cast< float_t >( grid4Rows.size()) };

				// run grid of 4 rows until it is out of frame
				int32_t offsetY4Rows { };
				while( ( offsetY4Rows + gridMaxY4Rows ) < 480 )
				{
					int32_t offsetX4Rows { };
					while( ( offsetX4Rows + gridMaxX4Rows ) < 752 )
					{
						int32_t goodPointsCount4Rows { };
						float_t avgPointDist4Rows { };

						// compare grid to every center of polygons
						for( const auto& gridPoint : grid4Rows )
						{
							for( size_t i = 0; i < polyMoments.size(); ++i )
							{
								// if center is in 40x40 range of grid
								if( gridPoint.x + offsetX4Rows + 50 > polyMoments[i].x &&
									polyMoments[i].x > gridPoint.x + offsetX4Rows - 50 &&
									gridPoint.y + offsetY4Rows + 50 > polyMoments[i].y &&
									polyMoments[i].y > gridPoint.y + offsetY4Rows - 50 )
								{
									++goodPointsCount4Rows;
									avgPointDist4Rows += sqrtf( powf( gridPoint.x + offsetX4Rows - polyMoments[i].x, 2.0f ) +
									                            powf( gridPoint.y + offsetY4Rows - polyMoments[i].y, 2.0f ));
								}
							}
						}

						// calculate best average dist between grid points and centers to get the best grid
						avgPointDist4Rows /= static_cast< float_t >( goodPointsCount4Rows );
						float_t ratio4Rows { static_cast< float_t >( goodPointsCount4Rows ) /
						                     static_cast< float_t >( grid4Rows.size() ) };

						// update datas of best grid found
						if( bestRatio4Rows <= ratio4Rows &&
							ratio4Rows >= minRatio4Rows &&
							avgPointDist4Rows <= bestAvgPointDist4Rows )
						{
							bestRatio4Rows = ratio4Rows;
							bestOffsetX4Rows = offsetX4Rows;
							bestOffsetY4Rows = offsetY4Rows;
							bestAvgPointDist4Rows = avgPointDist4Rows;
						}
						offsetX4Rows += 5;
					}
					offsetY4Rows += 5;
				}

				//==================================================================================
				// 3 ROWS

				// find farthest point of grid of 3 rows
				int32_t gridMaxX3Rows { };
				int32_t gridMaxY3Rows { };
				for( size_t i = 0; i < grid3Rows.size(); ++i )
				{
					if( grid3Rows[i].x > gridMaxX3Rows )
					{
						gridMaxX3Rows = grid3Rows[i].x;
					}
					if( grid3Rows[i].y > gridMaxY3Rows )
					{
						gridMaxY3Rows = grid3Rows[i].y;
					}
				}

				int32_t bestOffsetX3Rows { };
				int32_t bestOffsetY3Rows { };
				float_t bestRatio3Rows { };
				float_t bestAvgPointDist3Rows { 100.0f };
				float_t minRatio3Rows { static_cast< float_t >( grid3Rows.size() - 2 ) /
				                        static_cast< float_t >( grid3Rows.size() ) };

				// run grid of 3 rows until it is out of frame
				int32_t offsetY3Rows { };
				while( ( offsetY3Rows + gridMaxY3Rows) < 480 )
				{
					int32_t offsetX3Rows { };
					while( ( offsetX3Rows + gridMaxX3Rows) < 752 )
					{
						int32_t goodPointsCount3Rows { };
						float_t avgPointDist3Rows { };

						// compare grid to every center of polygons
						for( const auto& gridPoint : grid3Rows )
						{
							for( size_t i = 0; i < polyMoments.size(); ++i )
							{
								// if center is in 40x40 range of grid
								if( gridPoint.x + offsetX3Rows + 50 > polyMoments[i].x &&
								    polyMoments[i].x > gridPoint.x + offsetX3Rows - 50 &&
								    gridPoint.y + offsetY3Rows + 50 > polyMoments[i].y &&
								    polyMoments[i].y > gridPoint.y + offsetY3Rows - 50 )
								{
									++goodPointsCount3Rows;
									avgPointDist3Rows += sqrtf( powf( gridPoint.x + offsetX3Rows - polyMoments[i].x, 2.0f ) +
									                            powf( gridPoint.y + offsetY3Rows - polyMoments[i].y, 2.0f ));
								}
							}
						}

						// calculate best average dist between grid points and centers to get best grid
						avgPointDist3Rows /= static_cast< float_t >(goodPointsCount3Rows);
						float_t ratio3Rows { static_cast< float_t >(goodPointsCount3Rows) /
						                     static_cast< float_t >(grid3Rows.size()) };

						// update datas of best grid found
						if( bestRatio3Rows <= ratio3Rows &&
							ratio3Rows >= minRatio3Rows &&
							avgPointDist3Rows <= bestAvgPointDist3Rows )
						{
							bestRatio3Rows = ratio3Rows;
							bestOffsetX3Rows = offsetX3Rows;
							bestOffsetY3Rows = offsetY3Rows;
							bestAvgPointDist3Rows = avgPointDist3Rows;
						}
						offsetX3Rows += 3;
					}
					offsetY3Rows += 3;
				}

				// display grid if ratio is good enough
				if( bestRatio4Rows > minRatio4Rows )
				{
					cl::print_line( "Best offset: ", bestOffsetX4Rows, ",", bestOffsetY4Rows );
					cl::print_line( "Ratio: ", bestRatio4Rows );
					cl::print_line( "Avg dist: ", bestAvgPointDist4Rows );

					for( size_t i = 0; i < grid4Rows.size(); ++i )
					{
						cv::rectangle( src_cpy,
						               cv::Point( grid4Rows[i].x + bestOffsetX4Rows - 50,
						                          grid4Rows[i].y + bestOffsetY4Rows - 50 ),
						               cv::Point( grid4Rows[i].x + bestOffsetX4Rows + 50,
						                          grid4Rows[i].y + bestOffsetY4Rows + 50 ),
						               scalar_red );
					}

					// erase out of grid hulls
					hullPointsIt = hullPoints.begin();
					while( hullPointsIt != hullPoints.end() )
					{
						cv::Moments m = moments( (*hullPointsIt), false );
						cv::Point2i pt( cl::math::RoundToLong( m.m10 / m.m00 ),
						                cl::math::RoundToLong( m.m01 / m.m00 ));

						bool isInBestGrid { };
						for( const auto& gridPoint : grid4Rows )
						{
							if( gridPoint.x + bestOffsetX4Rows + 50 > pt.x &&
								pt.x > gridPoint.x + bestOffsetX4Rows - 50 &&
								gridPoint.y + bestOffsetY4Rows + 50 > pt.y &&
								pt.y > gridPoint.y + bestOffsetY4Rows - 50 )
							{
								isInBestGrid = true;
							}
						}

						if( !isInBestGrid )
						{
							std::vector< std::vector< cv::Point > > tmp;
							tmp.push_back((*hullPointsIt));
							cv::drawContours( src_cpy, tmp, -1, scalar_red );
							hullPointsIt = hullPoints.erase( hullPointsIt );
						}
						else
						{
							++hullPointsIt;
						}
					}
				}
				else if( bestRatio3Rows > minRatio3Rows )
				{
					cl::print_line( "Best offset: ", bestOffsetX3Rows, ",", bestOffsetY3Rows );
					cl::print_line( "Ratio: ", bestRatio3Rows );
					cl::print_line( "Avg dist: ", bestAvgPointDist3Rows );

					for( size_t i = 0; i < grid3Rows.size(); ++i )
					{
						cv::rectangle( src_cpy,
						               cv::Point( grid3Rows[i].x + bestOffsetX3Rows - 50,
						                          grid3Rows[i].y + bestOffsetY3Rows - 50 ),
						               cv::Point( grid3Rows[i].x + bestOffsetX3Rows + 50,
						                          grid3Rows[i].y + bestOffsetY3Rows + 50 ),
						               scalar_red );
					}

					// erase out of grid hulls
					hullPointsIt = hullPoints.begin();
					while( hullPointsIt != hullPoints.end() )
					{
						cv::Moments m = moments( (*hullPointsIt), false );
						cv::Point2i pt( cl::math::RoundToLong( m.m10 / m.m00 ),
						                cl::math::RoundToLong( m.m01 / m.m00 ));

						bool isInBestGrid{ };
						for( const auto& gridPoint : grid3Rows )
						{
							if( gridPoint.x + bestOffsetX3Rows + 50 > pt.x &&
								pt.x > gridPoint.x + bestOffsetX3Rows - 50 &&
								gridPoint.y + bestOffsetY3Rows + 50 > pt.y &&
								pt.y > gridPoint.y + bestOffsetY3Rows - 50 )
							{
								isInBestGrid = true;
							}
						}

						if( !isInBestGrid )
						{
							std::vector< std::vector< cv::Point > > tmp;
							tmp.push_back( (*hullPointsIt) );
							cv::drawContours(src_cpy, tmp, -1, scalar_red );
							hullPointsIt = hullPoints.erase( hullPointsIt );
						}
						else
						{
							++hullPointsIt;
						}
					}
				}
				else
				{
					cl::print_line( "No good match for grid" );
					hullPoints.resize( 0 );
				}
				//
				int32_t goodPointsCount{ };
				while( grid3Rows[8].y > 480 )
				{
					for( const auto& moment : polyMoments )
					{
						for( const auto& gridPoint : grid3Rows )
						{
							if( gridPoint.x + 20 > moment.x && moment.x > gridPoint.x - 20 &&
							    gridPoint.y + 20 > moment.y && moment.y > gridPoint.y - 20 )
							{
								++goodPointsCount;
							}
						}
					}
					auto gridIt = grid3Rows.begin();
					while( gridIt != grid3Rows.end() )
					{
						(*gridIt).y += 140;
					}
				}
				//
				//auto listItX = listAlignedPointsX.begin();
				//while( listItX != listAlignedPointsX.end() )
				//{
				//	int32_t avgX{ };
				//	for( const auto& pt : (*listItX) )
				//	{
				//		avgX += pt.x;
				//	}
				//
				//	avgX /= (*listItX).size();
				//	cv::line( src_cpy, cv::Point( avgX, 0 ), cv::Point( avgX, 480 ), scalar_red );
				//	cv::line( src_cpy, cv::Point( avgX - 30, 0 ), cv::Point( avgX - 30, 480 ), scalar_red );
				//	cv::line( src_cpy, cv::Point( avgX + 30, 0 ), cv::Point( avgX + 30, 480 ), scalar_red );
					//++listItX;
				//}

				//auto listItY = listAlignedPointsY.begin();
				//while( listItY != listAlignedPointsY.end())
				//{
				//	int32_t avgY{ };
				//	for( const auto& pt : (*listItY))
				//	{
				//		avgY += pt.y;
				//	}
				//
				//	avgY /= (*listItY).size();
				//	cv::line( src_cpy, cv::Point( 0, avgY ), cv::Point( 752, avgY ), scalar_red );
					//cv::line( src_cpy, cv::Point( 0, avgY - 30 ), cv::Point( 752, avgY - 30), scalar_red );
					//cv::line( src_cpy, cv::Point( 0, avgY + 30 ), cv::Point( 752, avgY + 30), scalar_red );
					//++listItY;
				//}



				cv::Mat src_cpy2 = sourceL.clone();
				// draw hull contours on temp image (hulls empty otherwise for some reason)
				if( hullPoints.size())
					cv::drawContours( src_cpy2, hullPoints, -1, scalar_yellow );
				// draw keypoints on temp image (keypoints empty otherwise for some reason)
				if( keyPoints.size())
					cv::drawKeypoints( src_cpy2, keyPoints, intermediate,
					                   scalar_white,
					                   cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

				/*//*
				//==============================================================
				// Kalman

				// save hulls which don't touch the top or bottom edge
				std::vector< PolygonPtr > hullPoly;
				for( size_t i = 0; i < hullPoints.size(); ++i )
				{
					PolygonPtr polygonPtr( std::make_shared< Polygon >());
					polygonPtr->initialize( hullPoints[i] );
					if( !polygonPtr->is_touching_edge())
					{
						polygonPtr->draw_min_ellipse( intermediate, scalar_green );
						polygonPtr->draw_contour( intermediate, scalar_green );
						hullPoly.push_back( polygonPtr );
					}
				}

				// erase lost objects
				trackedObjects.erase( std::remove_if( trackedObjects.begin(),
				                                       trackedObjects.end(),
				                                       [ ](
					                                       ObjectTrackerPtr obj )
				                                       {
				                                           return
					                                           obj->lost_count_ >
					                                           15;
				                                       } ),
				                       trackedObjects.end());

				// Update prediction for all tracked objects
				for( const auto& tracked : trackedObjects )
				{
					tracked->compute_prediction();
					tracked->has_polygon_ = false;
				}

				// track with polys

				// used to extend bounding rects in case the tracked object
				// in case it goes out of bounds (usually in the begining of tracking)
				int32_t extension{ 30 };
				cv::Point2i estimated;

				auto polyIt = hullPoly.begin();
				while( polyIt != hullPoly.end() )
				{
					bool alreadyTracked{};

					// extend bounding rect of hull
					cv::Rect boundingRectExtended = (*polyIt)->bound_rect();
					if( boundingRectExtended.x > extension )
						boundingRectExtended.x -= extension;
					if( boundingRectExtended.y > extension )
						boundingRectExtended.y -= extension;
					boundingRectExtended.height += extension * 2;
					boundingRectExtended.width += extension * 2;

					cv::rectangle(wb, boundingRectExtended, scalar_red);

					for( auto& tracked : trackedObjects )
					{
						// Correct and draw prediction
						if( tracked->is_in_bouding_rect( boundingRectExtended ) )
						{
							estimated = tracked->get_estimated_position( (*polyIt)->moment() );

							tracked->update_mean_ellipse( (*polyIt)->min_ellipse(), estimated );
							//tracked->draw_ellipse( sourceL, scalar_green );
							//
							//cv::circle( sourceL, estimated, 2, scalar_green, -1,
							//            CV_AA );
							//putText( sourceL, std::to_string( tracked->index_ ),
							//         estimated,
							//         CV_FONT_NORMAL, 0.5, scalar_white, 1,
							//         CV_AA );
							tracked->has_polygon_ = true;
							tracked->lost_count_ = 0;
							alreadyTracked = true;
							break;
						}
					}

					// if poly is not already tracked, track it
					if( !alreadyTracked )
					{
						ObjectTrackerPtr newt = std::make_shared< ObjectTracker >();
						trackedObjects.push_back( newt );

						newt->initialize( (*polyIt)->moment(), ++index );

						newt->has_polygon_ = true;
						estimated = newt->get_estimated_position( (*polyIt)->moment() );

						newt->update_mean_ellipse( (*polyIt)->min_ellipse(), estimated );
						//newt->draw_ellipse( sourceL, scalar_green );
						//
						//cv::circle( sourceL, estimated, 2, scalar_green, -1, CV_AA );
						//putText( sourceL, std::to_string( newt->index_ ), estimated,
						//         CV_FONT_NORMAL, 0.5, scalar_white, 1, CV_AA );
					}

					polyIt = hullPoly.erase( polyIt );
				}

				for( const auto& tracked : trackedObjects )
				{
					if( tracked->has_polygon() )
					{
						estimated = tracked->get_position();
						tracked->draw_ellipse( sourceL, scalar_green );

						cv::circle( sourceL, estimated, 2, scalar_green, -1,
						            CV_AA );
						putText( sourceL, std::to_string( tracked->index_ ),
						         estimated,
						         CV_FONT_NORMAL, 0.5, scalar_white, 1,
						         CV_AA );
					}
					else
					{
						++tracked->lost_count_;
					}
				}

				// Kalman end
				//==============================================================
				/*//*//*

				//std::vector<Contour> raw_contours_resampled( raw_contours.size() );
				//for( size_t i = 0; i < raw_contours.size(); ++i )
				//{
				//	ResampleCurve( raw_contours[i], raw_contours_resampled[i], 200, false );
				//}

				//std::vector< std::vector< int > > hull( raw_contours.size() );
				//std::vector< std::vector< cv::Vec4i > > convDef( raw_contours.size() );
				//std::vector< std::vector< cv::Point > > hull_points( raw_contours.size() );
				//std::vector< std::vector< cv::Point > > defect_points( raw_contours.size() );
				//
				//std::vector< PolygonPtr > valid_poly;
				//std::vector< PolygonPtr > possible_valid_poly;
				//std::vector< PolygonPtr > invalid_poly;
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
				//cv::cvtColor( result_ab, wb, CV_GRAY2BGR );

				//cv::imshow( "L", imgL );
				//cv::imshow( "A", imgA );
				//cv::imshow( "B", imgB );
				//cv::imshow( "AInv", imgAInv );
				//cv::imshow( "AInvEq", imgAInvEq );
				//cv::imshow( "Thres", imgThres );
				//cv::imshow( "Eroded", imgEroded );
				//cv::imshow( "Dilated", imgDilated );
				//cv::imshow( "Final", imgDilated );

				// Haar cascades
				//cv::Mat src;
				//src = cv::imread( filePathL );
				//cv::Mat src_gray;
				//cv::cvtColor(src, src_gray, CV_BGR2GRAY);
				//
				//cv::CascadeClassifier cascadeClassifier;
				//if( !cascadeClassifier.load( "/home/robin/Documents/cascades/res/cascade.xml") )
				//	cl::print_line("Classifier not loaded");
				//std::vector< cv::Rect > objects;
				//cascadeClassifier.detectMultiScale( src_gray, objects,
				//                                    1.1, 2, 0 | CV_HAAR_SCALE_IMAGE,
				//                                    cv::Size( 30, 30 ));
				//for( size_t i = 0; i < objects.size(); ++i )
				//{
				//	cv::rectangle(src, objects[i], scalar_red);
				//}
				//cl::print_line(objects.size());
				//cv::imshow("Cascade", src);

			//cv::imshow( "src", img );
			//cv::imshow( "out", salad_detector.get_output_img() )
			};

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

			//cv::Mat labSplitOut( size.height, size.width * 3, CV_8UC3 );
			//cv::Mat imgL(labSplitOut, cv::Rect(0, 0, size.width, size.height));
			//vectLab2[0].copyTo(imgL);
			//cv::Mat imgA(labSplitOut, cv::Rect(0, size.width, size.width, size.height));
			//vectLab2[1].copyTo(imgA);
			//cv::Mat imgB(labSplitOut, cv::Rect(0, size.width * 2, size.width, size.height));
			//vectLab2[2].copyTo(imgB);

			//std::string folderToVid { "/home/robin/Programming/ObjectTracker/vid/" };
			//folderToVid.append(std::to_string(image_nbr));
			//folderToVid.append("_out.tiff");
			//cv::imwrite(folderToVid, final);

			cv::imshow( "final_output", combined );


			//--------------------------------------------------------
			//------------------------TESTS---------------------------
			//--------------------------------------------------------
			//int lowHue;
			//int highHue;
			//int lowSat;
			//int highSat;
			//int lowVal;
			//int highVal;
			//
			//cv::Mat imgDst;
			//cv::Mat imgSrcRGB = cv::Mat( size, CV_8UC3 );
			//imgSrcRGB = cv::imread( filePathL );
			//
			//cv::createTrackbar("Low Hue", "tests", &lowHue, 179);
			//cv::createTrackbar("High Hue", "tests", &highHue, 179);
			//cv::createTrackbar("Low Sat", "tests", &lowSat, 255);
			//cv::createTrackbar("High Sat", "tests", &highSat, 255);
			//cv::createTrackbar("Low Val", "tests", &lowVal, 255);
			//cv::createTrackbar("High Val", "tests", &highVal, 255);
			//
			//cv::Mat imgSrcHSV;
			//cv::cvtColor(imgSrcRGB, imgSrcHSV, CV_BGR2HSV);
			//
			//cv::inRange(imgSrcHSV, cv::Scalar(lowHue, lowSat, lowVal), cv::Scalar(highHue, highSat, highVal), imgDst);
			//
			//// morph opening (remove small object from foreground)
			//cv::erode( imgDst, imgDst,
			//           cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)) );
			//cv::dilate( imgDst, imgDst,
			//            cv::getStructuringElement( cv::MORPH_ELLIPSE, cv::Size( 5, 5 )) );
			//
			//// morph closing (fill small holes in foreground)
			//cv::dilate( imgDst, imgDst,
			//            cv::getStructuringElement( cv::MORPH_ELLIPSE, cv::Size( 5, 5 )) );
			//cv::erode( imgDst, imgDst,
			//           cv::getStructuringElement( cv::MORPH_ELLIPSE, cv::Size( 5, 5 )) );

			//cv::imshow("tests", imgDst);

			//--------------------------------------------------------
			//--------------------------------------------------------
			//--------------------------------------------------------*/

			key_pressed = static_cast<uint8_t>( cv::waitKey( 10 ) );
			if( key_pressed == static_cast<uint8_t>('r') )
			{
				need_hist_reset = true;
			}
		}

		//camera->close();
		ht::log_info( "Exiting main loop" );
	};

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
