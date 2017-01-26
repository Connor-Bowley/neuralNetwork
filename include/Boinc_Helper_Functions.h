
#ifndef __BOINC_HELPER_FUNCTIONS__
#define __BOINC_HELPER_FUNCTIONS__

#ifdef _BOINC_APP_
	#include "diagnostics.h"
	#include "filesys.h"
	#include "boinc_api.h"
	#include "mfile.h"
	#include "proc_control.h"
#endif

#include <string>

//BOINC FUNCTIONS
std::string getBoincFilename(std::string filename) throw(std::runtime_error) {
    std::string resolved_path = filename;
	#ifdef _BOINC_APP_
	    if(boinc_resolve_filename_s(filename.c_str(), resolved_path)) {
	        printf("Could not resolve filename %s\n",filename.c_str());
	        throw std::runtime_error("Boinc could not resolve filename");
	    }
	#endif
    return resolved_path;
}

#endif