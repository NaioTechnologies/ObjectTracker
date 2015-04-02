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

#ifndef ENTRYPOINT_H
#define ENTRYPOINT_H

//=================================================================================================
// I N C L U D E   F I L E S

#include "HTCmdLineParser.h"
#include "HTLogger.h"


//=================================================================================================
// F O R W A R D   D E C L A R A T I O N S

//=================================================================================================
// C O N S T A N T S

//=================================================================================================
// C L A S S E S

class EntryPoint
{
//--Methods----------------------------------------------------------------------------------------
public:
	EntryPoint();

	~EntryPoint();

	/// Outputs a header for the program on the standard output stream.
	void print_header() const;

	bool handle_parameters( const std::string& option, const std::string& type );

	/// Main entry point of the application.
	int32_t run( int argc, const char** argv );

	static void call_back_func( int event, int x, int y, int flags, void* userdata );

//--Data members-----------------------------------------------------------------------------------
private:
	/// Command line parser, logger and debugger for the application
	HTCmdLineParser parser_;
	HTCmdLineParser::Visitor handler_;

	bool pressed_;

	cv::Point2i ul_;
	cv::Point2i dr_;
};


//=================================================================================================
// I N L I N E   F U N C T I O N S   C O D E   S E C T I O N

#endif  // ENTRYPOINT_H
