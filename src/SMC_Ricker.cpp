#include "RcppSMC.h"
#include <cmath>

using namespace std;

long lIterates_outer;
long lIterates_inner;

class Ricker{
public:
	arma::vec theta;
	double loglike, logprior;
};

smc::sampler<double,arma::vec> * Sampler_inner;
arma::vec temps;
arma::vec y;

arma::mat boundaries(3,2);

double logLikelihood(const Ricker & current);
double logPrior(const Ricker & current);
void fInitialise_outer(Ricker & current, double & logweight, smc::nullParams & param);
void fMove_outer(long lTime, Ricker & current, double & logweight, smc::nullParams & param);
bool fMCMC_outer(long lTime, Ricker & current, double & logweight, smc::nullParams & param);
void fInitialise_inner(double & X, double & logweight, arma::vec & param);
void fMove_inner(long lTime, double & X, double & logweight, arma::vec & param);

//' @param data A vector of observed data
//' @param outN The number of particles in likelihood annealing SMC (the outer level of SMC)
//' @param inN The number of particles to use in the particle filtering run to estimate the likelihood
//' @param temperatures The likelihood annealing temperature schedule
//' 
//' @examples
//' \dontrun{ 
//' # some data
//' y <- c(93,1,34,100,0,8,264,0,0,0,
//'      0,2,44,15,164,0,0,1,60,1,
//'      70,5,215,0,0,0,1,12,189,0,
//'      0,0,6,144,0,0,12,182,0,0,
//'      1,17,123,0,15,97,0,13,274,0)
//' 		
//' # Performing the SMC run with N1 particles in likelihood-annealing SMC and
//' # N2 particles used in the particle filtering estimates of the likelihood
//' N1 <- 1000
//' N2 <- 1000
//' results <- SMC_Ricker(y,N1,N2,seq(0,1,0.1)^5)
//' 			
//' # Getting the posterior means
//' myMeans <- colSums(results$samples*matrix(rep(results$weights,3),nrow=N1,byrow=TRUE))
//' myMeans
//' 				
//' # Log of the estimated normalising constant (for model choice)
//' results$logZ
//' 					
//' # Posterior density marginal plots
//' true_logged <- c(3.8,2.3026,-1.204)
//' res <- as.data.frame(results)
//' require(ggplot2)
//' ggplot(res, aes(samples.1)) + geom_density(aes(weight=weights)) +
//' 	geom_vline(aes(xintercept = true_logged[1])) + xlab("log r")
//' ggplot(res, aes(samples.2)) + geom_density(aes(weight=weights)) +
//' 	geom_vline(aes(xintercept = true_logged[2])) + xlab("log phi")
//' ggplot(res, aes(samples.3)) + geom_density(aes(weight=weights)) +
//' 	geom_vline(aes(xintercept = true_logged[3])) + xlab("log sigma")
//'}
//' @name RickerExample-package
// [[Rcpp::export]]
Rcpp::List SMC_Ricker(arma::vec data, unsigned long outN, unsigned long inN, arma::vec temperatures) {
	
	try {
		y = data;
		
		temps = temperatures;
		
		lIterates_inner = data.n_rows;
		lIterates_outer = temps.n_rows;
		
		boundaries(0,0) = 2; boundaries(0,1) = 5;
		boundaries(1,0) = 1.61; boundaries(1,1) =3;
		boundaries(2,0) = -3; boundaries(2,1) = -0.22;
		
		// The inner sampler
		Sampler_inner = new smc::sampler<double,arma::vec>(inN, HistoryType::NONE);
		smc::moveset<double,arma::vec> Moveset_inner(fInitialise_inner, fMove_inner, NULL);
		Sampler_inner->SetResampleParams(ResampleType::MULTINOMIAL, inN);
		Sampler_inner->SetMoveSet(Moveset_inner);
		
		// The outer sampler
		smc::sampler<Ricker,smc::nullParams> Sampler_outer(outN, HistoryType::NONE);
		smc::moveset<Ricker,smc::nullParams> Moveset_outer(fInitialise_outer, fMove_outer, fMCMC_outer);
		Sampler_outer.SetResampleParams(ResampleType::MULTINOMIAL, 0.5);
		Sampler_outer.SetMoveSet(Moveset_outer);
		Sampler_outer.SetMcmcRepeats(10);
		
		// Running the main sampler
		Sampler_outer.Initialise();
		//Sampler_outer.IterateUntil(lIterates_outer-1);
		for(int n=1; n < lIterates_outer; ++n){
		  Sampler_outer.Iterate();
		  
		  Rcpp::Rcout << Sampler_outer.GetLogNCPath() << std::endl;
		}
		
		arma::mat theta(outN,3);
		arma::vec weights = Sampler_outer.GetParticleWeight();
		
		for (unsigned int i = 0; i<outN; i++){
			theta.row(i) = Sampler_outer.GetParticleValueN(i).theta.t();
		}
		
		double logNC = Sampler_outer.GetLogNCPath();
		
		delete Sampler_inner;
		
		return Rcpp::List::create(Rcpp::Named("samples") = theta,
                                 Rcpp::Named("weights") = weights,
                                 Rcpp::Named("logZ") = logNC);
	}
	catch(smc::exception  e) {
		Rcpp::Rcout << e;
	}
	return R_NilValue;            // to provide a return
}

double logLikelihood(const Ricker & current)
{
	Sampler_inner->SetAlgParam(current.theta);
	Sampler_inner->Initialise();
	Sampler_inner->IterateUntil(lIterates_inner-1);
	double logNC = Sampler_inner->GetLogNCPath();
	return logNC;
}

double logPrior(const Ricker & current)
{
	if ( (sum(current.theta>=boundaries.col(0))==3) && (sum(current.theta<=boundaries.col(1))==3) )
		return 0.0;
	else
		return -std::numeric_limits<double>::infinity();
}

void fInitialise_outer(Ricker & current, double & logweight, smc::nullParams & param)
{
	current.theta = boundaries.col(0) + (boundaries.col(1) - boundaries.col(0)) % Rcpp::as<arma::vec>(Rcpp::runif(3));
  current.logprior = logPrior(current);
	current.loglike = logLikelihood(current);
	logweight = current.logprior;
}

void fMove_outer(long lTime, Ricker & current, double & logweight, smc::nullParams & param)
{
	logweight += (temps(lTime) - temps(lTime-1))*current.loglike;
}

bool fMCMC_outer(long lTime, Ricker & current, double & logweight, smc::nullParams & param)
{
	Ricker value_prop;
	value_prop.theta = current.theta + 0.25*Rcpp::as<arma::vec>(Rcpp::rnorm(3)); // + cholCovRW*Rcpp::as<arma::vec>(Rcpp::rnorm(3));       
	value_prop.logprior = logPrior(value_prop);
	if (isinf(value_prop.logprior)==false){
	  value_prop.loglike = logLikelihood(value_prop);
	  
	  double MH_ratio = exp(temps(lTime)*(value_prop.loglike - current.loglike) + value_prop.logprior - current.logprior);
	  
	  if (MH_ratio>R::runif(0,1)){
	    current = value_prop;
	    return TRUE;
	  }
	}
	
	return FALSE;
}


void fInitialise_inner(double & X, double & logweight, arma::vec & param)
{
	double r = exp(param(0));
	double phi = exp(param(1));
	double sigmae = exp(param(2));
	X = r*exp(-1.0+sigmae*R::rnorm(0,1));
	double mu = phi*X; //lambda for poisson
	logweight = -lgamma(y(0)+1) - mu + y(0)*log(mu);
}

void fMove_inner(long lTime, double & X, double & logweight, arma::vec & param)
{
  double r = exp(param(0));
  double phi = exp(param(1));
  double sigmae = exp(param(2));
	X = r*X*exp(-X+sigmae*R::rnorm(0,1));
	double mu = phi*X; //lambda for poisson
	logweight += -lgamma(y(lTime)+1) - mu + y(lTime)*log(mu);
}

