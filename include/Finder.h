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

#ifndef FINDER_H
#define FINDER_H

//==================================================================================================
// I N C L U D E   F I L E S   A N D   F O R W A R D   D E C L A R A T I O N S

#include <opencv2/core/core.hpp>


//==================================================================================================
// C O N S T A N T S

//==================================================================================================
// C L A S S E S

class ObjectFinder
{
//--Methods-----------------------------------------------------------------------------------------
public:
	ObjectFinder();

	~ObjectFinder();

	// Sets the threshold on histogram values [0,1]
	void set_threshold( const float t );

	// Gets the threshold
	float get_threshold();

	// Sets the reference histogram
	void set_histogram( const cv::MatND& h );

	// Sets the reference histogram
	void set_histogram( const cv::SparseMat& h );

	// Finds the pixels belonging to the histogram
	cv::Mat find( const cv::Mat& image );

	cv::Mat find( const cv::Mat& image, float minValue, float maxValue, int* channels, int dim );

//--Data members------------------------------------------------------------------------------------
private:
	float hranges_[2];
	const float* ranges_[3];
	int32_t channels_[3];

	float threshold_;
	cv::MatND histogram_;
	cv::SparseMat shistogram_;
	bool isSparse_;

};


//==================================================================================================
// I N L I N E   F U N C T I O N S   C O D E   S E C T I O N

#endif
