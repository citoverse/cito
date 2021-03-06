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
> Maintainer: 'Christian Ames??der <Christian.Amesoeder@stud.uni-regensburg.de>'  
> New submission

This is expected

The devtools spell_check() gives out  some words, however all of them have been checked and are indeed correct.

The Github Windows-latest R-release fails with: 
> Error: Error: <callr_remote_error: Failed to build source package 'igraph'>  
> in process 4736   
> -->  
> Failed to build source package 'igraph', stdout + stderr:  

This seems to be a temporary external problem and I expect it to be gone in a couple of days. 


