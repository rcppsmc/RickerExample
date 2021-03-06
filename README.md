## RickerExample

This is an example of how to use RcppSMC in a package. I’m not sure whether this example is going to go
into the paper yet and it’s not fully documented, but it should give you some idea of how to create
another package that uses RcppSMC. It is fairly basic, using only a single function which is in C++ and no R functions.

At this stage, you will need to use the Github version of RcppSMC to be able to access RcppSMC.h. After
we update CRAN (soon), this won't be required.

### How to build the package
You can build the package using

```{r, eval = FALSE}
library('devtools')
install_github("rcppsmc/rcppsmc")
Rcpp::compileAttributes("RickerExample")
devtools::document("RickerExample") # if you have Roxygen documentation
devtools::load_all('RickerExample')
devtools::build('RickerExample')
```

and look at the documentation with `?RickerExample`. There is some example code in the documentation, but you may wish to reduce `N1` and `N2` to 100 for a quick run.

### Using RcppSMC without creating your own package

If you wanted to use RcppSMC without creating your own package, you can do so using \code{sourceCpp}.


Simply add the line `//[[Rcpp::depends(RcppSMC)]]` to the top of your C++ file, SMC_Ricker.cpp and replace `library(RickerExample)` in the example code from the documentation with

```{r, eval = FALSE}
#devtools::install_github("rcppsmc/rcppsmc")
library(RcppSMC)
Rcpp::sourceCpp('SMC_Ricker.cpp')
```