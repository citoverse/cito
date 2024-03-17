# Version 1.1
## cito submission 1 2024/03/17
This is a major update. The update includes new loss/likelihood functions, 
hyperparameter tuning, bug fixes, and improved documentation.

Best,
Max Pichler

### Successfull R CMD checks under

Locally: 
* MacOS Monterey 14.1.2 (R aarch64-apple-darwin20) (CPU and mps devices)
* Ubuntu 20.04 (for GPU device)

Github actions:
* macOS-latest (release)
* Ubuntu 20.04 R-release, and R-development
* Windows-latest R-release

Win-builder 
* R-release, R-development, and R-oldrelease

Spelling has been checked (potential errors are intended).



# Version 1.0.2
## cito submission 5 2023/06/10
Dear CRAN Team,

I apologize for the inconvenience (several submission attempts), but it seems that
that something went wrong with my last attempt? 

All issues were addressed. 

Best,
Maximilian Pichler

## cito submission 5 2023/01/10
Issue:
Unfortunately, the manual cannot be built on the CRAN ,aster with its
older LaTeX installtion and hence this never got published.
Reason:
LaTeX Warning: File `learning\_rates.jpg' not found on input line 881.
! Package pdftex.def Error: File `learning\T1\textunderscorerates.jpg'
not found
d: using draft setting.
See the pdftex.def package documentation for explanation.
Type H for immediate help.
...
l.881 \Figure{learning\_rates.jpg}{}
Can you please rename the file and omit the underscore? ALso in the Rd
files that refer to it.

Done. Thank you.

Best,
Maximilian Pichler



## cito submission 4 2023/29/9
Issues:
Found the following (possibly) invalid DOIs:
DOI: arXiv.2303.09599
From: inst/CITATION
Message: Invalid DOI

Last time I wrote

"... please onlk use the DOI, i.e.
10.48550/arXiv.2303.09599
within the DOI markup."

Please fix and resubmit.


Fixed, thanks.

Best,
Maximilian Pichler



## cito submission 3 2023/29/9
Issues:

Found the following (possibly) invalid DOIs:
DOI: https://doi.org/10.48550/arXiv.2303.09599
From: inst/CITATION
Status: 404
Message: Not Found

This is an URL, please onlky use the DOI, i.e. 10.48550/arXiv.2303.09599
within the DOI markup.


Size of tarball: 12294996 bytes

Both issues were addressed. The package should have now a size of less than 5MB.

Best,
Maximilian Pichler



## cito submission 2 2023/29/9
This is a major update where several features such as bootstrapping for uncertainties and additional xAI methods 
(average conditional effects) have been added. Also, several bugs have been fixed, documentation has been improved,
and 3 vignettes were added.

We would also like to announce a change of maintainer: Maximilian Pichler, co-author of cito, 
will take over the position of maintainer.

Old maintainer: 
Christian Amesöder, Christian.Amesoeder@informatik.uni-regensburg.de

New Maintainer:
Maximilian Pichler, maximilian.pichler@biologie.uni-regensburg.de

In the first submission we had problems passing the rcmdcheck. We changed the vignettes back to .rmd files.

Best,
Maximilian Pichler

### Successfull R CMD checks under

Locally: MacOS Monterey 13.4.1 (R aarch64-apple-darwin20)

Github actions:
* macOS-latest (release)
* Ubuntu 20.04 R-release, and R-development
* Windows-latest R-release

Win-builder 
* R-release, R-development, and R-oldrelease


## cito submission 1 2023/22/9
This is a major update where several features such as bootstrapping for uncertainties and additional xAI methods 
(average conditional effects) have been added. Also, several bugs have been fixed, documentation has been improved,
and 3 vignettes were added.

We would also like to announce a change of maintainer: Maximilian Pichler, co-author of cito, 
will take over the position of maintainer.

Old maintainer: 
Christian Amesöder, Christian.Amesoeder@informatik.uni-regensburg.de

New Maintainer:
Maximilian Pichler, maximilian.pichler@biologie.uni-regensburg.de

Due to the long creation time of the vignettes, we decided to include them as 
pre-built static HTML vignettes using the R.rsp package. 
We hope that we have included them in the right way. 

Best,
Max

### Successfull R CMD checks under

Locally: MacOS Monterey 13.4.1 (R aarch64-apple-darwin20)

Github actions:
* macOS-latest (release)
* Ubuntu 20.04 R-release, and R-development
* Windows-latest R-release

Win-builder 
* R-release, R-development, and R-oldrelease

Notes from win-builder:
Found the following hidden files and directories:
  vignettes/.html
These were most likely included in error. See section 'Package
structure' in the 'Writing R Extensions' manual.





# Version 1.0.1
## cito submission 1 2022/03/10
This is a minor update, fixing a few bugs in the ALE and PDP functions, 
and improving the runtime by changing the dataloader
Maintainer E-mail updated from *@stud.uni-regensburg.de to @informatik.uni-regensburg.de

### Successfull R CMD checks under



# Version 1.0.0
## cito submission 3  2022/07/22

There was an issue with the tests, which Brian Ripley kindly found out and took of CRAN. 
This issue is resolved and the tests as well as examples have been reworked.

Sorry for the inconvenience 

Cheers,
Christian

## cito submission 2 2022/07/18
Hello, 

Thanks for the feedback, I adressed all your points below.

Best, Christian

>Please always write package names, software names and API (application programming interface) names in single quotes in title and description.
e.g: --> 'R'
Please note that package names are case sensitive.

Done.


>If there are references describing the methods in your package, please add these in the description field of your DESCRIPTION file in the form authors (year) <doi:...> authors (year) <arXiv:...> authors (year, ISBN:...) or if those are not available: <https:...> with no space after 'doi:', 'arXiv:', 'https:' and angle brackets for auto-linking.
(If you want to add a title as well please put it in quotes: "Title")

There is no publication yet.  

>Please add \value to .Rd files regarding exported methods and explain the functions results in the documentation. Please write about the structure of the output (class) and also what the output means.
(If a function does not return a value, please document that too, e.g.
\value{No return value, called for side effects} or similar) Missing Rd-tags:
      ALE.Rd: \value
      PDP.Rd: \value
      plot.citodnn.Rd: \value
      print.summary.citodnn.Rd: \value
      
Done.     
      
>Please unwrap the examples if they are executable in < 5 sec, or replace \dontrun{} with \donttest{}.

Done, replaced \\dontrun{} with \\donttest{}. 

>Please do not install packages in your functions, examples or vignette.
This can make the functions, examples and cran-check very slow.

Done. 


## R CMD check results
   Successfull R CMD checks under
* Locally: Windows 11 (R x86_64 version)
* Github actions:
  - MacOS -latest R-release
  - Ubuntu 20.04 R-release, R-oldrelease, and R-development
  
* Rhub:
  - Windows Server 2022, R-devel, 64 bit
* Win-builder R-release, R-development, and R-oldrelease


1 Note for all Win-builder logs: 

> checking CRAN incoming feasibility ... NOTE  
> Maintainer: 'Christian Amesöder <Christian.Amesoeder@stud.uni-regensburg.de>'  
> New submission

This is expected

The devtools spell_check() gives out  some words, however all of them have been checked and are indeed correct.

The Github Windows-latest R-release fails with: 
> Error: Error: <callr_remote_error: Failed to build source package 'igraph'>  
> in process 4736   
> -->  
> Failed to build source package 'igraph', stdout + stderr:  

This seems to be a temporary external problem and I expect it to be gone in a couple of days. 


