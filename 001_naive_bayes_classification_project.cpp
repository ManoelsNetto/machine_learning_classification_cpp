// Classification model with Naive Bayes to categorize Digital signature

// Warning suppression
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"

// Libs
#include <iostream>
#include <chrono>
#include <fstream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <math.h>

// Namespace
using namespace std;
using namespace std::chrono;

// Start index - test dataset
const int startTest = 900;

// Helper Methods
vector<vector<double>> priorProb(vector<double> vect);
void print2DVector(vector<vector<double>> vect);
vector<vector<double>> countClass(vector<double> vect);
vector<vector<double>> likelihoodDoc_type(vector<double> classification, vector<double> doc_type, vector<vector<double>> count_classification);
vector<vector<double>> likelihoodValidDigitalSign(vector<double> classification, vector<double> valid_digitalsign, vector<vector<double>> count_valid_digitalsign);
vector<vector<double>> use_daysMean(vector<double> classification, vector<double> use_days, vector<vector<double>> count_use_days);
vector<vector<double>> use_daysVariance(vector<double> classification, vector<double> use_days, vector<vector<double>> count_use_days);
vector<vector<double>> use_daysMetrics(vector<vector<double>> use_daysMean, vector<vector<double>> use_daysVariance);
double calc_use_days_lh(double v, double mean_v, double var_v);

// Methods to implement the Bayes Theorem
vector<vector<double>> calc_raw_prob(double doc_type, double valid_digitalsign, double use_days, vector<vector<double>> prior, vector<vector<double>> lh_doc_type, vector<vector<double>> lh_valid_digitalsign, vector<vector<double>> use_days_mean, vector<vector<double>> use_days_var);

// Accuracy metrics
vector<vector<double>> confusionMatrix(vector<double> matA, vector<double> matB);
double accuracy(vector<double> matA, vector<double> matB);

// Number of forecasts
const int numOfIterations = 5;

// Main function
int main() {

	// Opening the file 
	fstream inputFile {"./data/dataset.csv"};
	
	// Checking if the file is open 
	if (!inputFile.is_open()) {

		cout << "Fail to open the file." << endl;

	};
	
	// Variables declaration
	
	// Scalar variables
	double idVal;
	double doc_typeVal;
	double classificationVal;
	double valid_digitalSignatureVal;
	double use_daysVal;
	
	// Vector variables
	vector<double> id;
	vector<double> doc_type;
	vector<double> classification;
	vector<double> valid_digitalSignature;
	vector<double> use_days;
	
	// Header variable
	string header;
	
	// Cell variable
	string cell;
	
	// getting the header line from the file to put in the header variable
	getline(inputFile, header);
	
	// Loading and cleaning data
	while (inputFile.good()) {
	
		// Reading Id column
		getline(inputFile, cell, ',');
		
		// Remove quotation marks
		cell.erase(remove(cell.begin(), cell.end(), '\"'), cell.end());
		
		// Reading non-empty cells
		if (!cell.empty()) {
		
			// String to double
			idVal = stod(cell);
			
			// Adding the id value at the end of the id vector
			id.push_back(idVal);
			
			// Reading doc_type
			getline(inputFile, cell, ',');
			
			// String to double
			doc_typeVal = stod(cell);
			
			// Adding the doc_type value at the end of the doc_type vector
			doc_type.push_back(doc_typeVal);
			
			// Reading classification
			getline(inputFile, cell, ',');
			
			// String to double
			classificationVal = stod(cell);
			
			// Adding the classification value at the end of the classification vector
			classification.push_back(classificationVal); 
			
			// Reading valid_digitalSignature
			getline(inputFile, cell, ',');
			
			// String to double
			valid_digitalSignatureVal = stod(cell);
			
			// Adding the valid_digitalSignature value at the end of the valid_digitalSignature vector
			valid_digitalSignature.push_back(valid_digitalSignatureVal);
			
			// Reading use_days
			getline(inputFile, cell);
			
			// String to double
			use_daysVal = stod(cell);
			
			// Adding the valid_digitalSignature value at the end of the valid_digitalSignature vector
			use_days.push_back(use_daysVal); 
		
		} else {
		
			// If cell is empty, then finish the while loop
			break;
		
		};
	
	};
	
	// Runtime
	auto start = high_resolution_clock::now();

	cout << "Runtime initiation" << endl << endl;
	
	// Training vectors
	
	// doc_type train set
	
	vector<double> doc_typeTrain;
	
	for (int i = 0; i < startTest; i++) {
	
		doc_typeTrain.push_back(doc_type.at(i));
	
	};
	
	// classification train set
	vector<double> classificationTrain;
	
	for (int i = 0; i < startTest; i++) {
	
		classificationTrain.push_back(classification.at(i));
	
	};

	// valid_digitalSignature train set
	vector<double> valid_digitalSignatureTrain;
	
	for (int i = 0; i < startTest; i++) {
	
		valid_digitalSignatureTrain.push_back(valid_digitalSignature.at(i));
	
	};

	// use_days train set
	vector<double> use_daysTrain;
	
	for (int i = 0; i < startTest; i++) {
	
		use_daysTrain.push_back(use_days.at(i));
	
	};

	// Test vectors
	
	// doc_type test set
	
	vector<double> doc_typeTest;
	
	for (int i = startTest; i < id.size(); i++) {
	
		doc_typeTest.push_back(doc_type.at(i));
	
	};
	
	// classification test set
	vector<double> classificationTest;
	
	for (int i = startTest; i < id.size(); i++) {
	
		classificationTest.push_back(classification.at(i));
	
	};

	// valid_digitalSignature test set
	vector<double> valid_digitalSignatureTest;
	
	for (int i = startTest; i < id.size(); i++) {
	
		valid_digitalSignatureTest.push_back(valid_digitalSignature.at(i));
	
	};

	// use_days test set
	vector<double> use_daysTest;
	
	for (int i = startTest; i < id.size(); i++) {
	
		use_daysTest.push_back(use_days.at(i));
	
	};
	
	// Naive bayes algorithm

	// Prior Probability
	// 1x2 matrix
	vector<vector<double>> prior = {priorProb(classificationTrain)};
	cout << "Prior Probability: " << endl;
	print2DVector(prior);
	cout << endl;

	// Vector to count the variable class
	// 1x2 matrix
	vector<vector<double>> count_class = {countClass(classificationTrain)};
	cout << "Prior Conditional" << endl;

	// Doc Type Likelihood (Probability)
	// 2x3 matrix
	vector<vector<double>> lh_doc_type = {likelihoodDoc_type(classificationTrain, doc_typeTrain, count_class)};
	cout << "\tDoc type" << endl;
	print2DVector(lh_doc_type);
	cout << endl;

	// Valid Digsign Likelihood (Probability)
	// 2x2 matrix
	vector<vector<double>> lh_valid_digital_sign = {likelihoodValidDigitalSign(classificationTrain, valid_digitalSignatureTrain, count_class)};
	cout << "\tvalid_digitalsign: " << endl;
	print2DVector(lh_valid_digital_sign);
	cout << "\n";

	// Mean and Variance of use_days variable
	// 1x2 matrix
	vector<vector<double>> use_days_mean = {use_daysMean(classificationTrain, use_daysTrain, count_class)};
	vector<vector<double>> use_days_var = {use_daysVariance(classificationTrain, use_daysTrain, count_class)};

	// use_days metrics
	cout << "\tuse_days: " << endl;
	vector<vector<double>> use_days_metrics = {use_daysMetrics(use_days_mean, use_days_var)};
	print2DVector(use_days_metrics);
	cout << endl << endl;

	// use_days mean
	cout << "use_days mean: " << endl;
	print2DVector(use_days_mean);
	cout << endl << endl;

	// use_days variance
	cout << "use_days variance: " << endl;
	print2DVector(use_days_var);
	cout << endl << endl;

	// Naive Bayes Algorithm ending
	auto stop = high_resolution_clock::now();

	// Vector to get the probabilities after the training
	// 1x2 matrix
	vector<vector<double>> raw(1, vector<double>(2,0));
	cout << "Predicting the probabilities of the test data: " << endl;

	// Lets get the 5 first predictions
	for (int i = startTest; i < (startTest + numOfIterations); i++) {

		// 1x2 matrix
		raw = calc_raw_prob(doc_type.at(i), valid_digitalSignature.at(i), use_days.at(i), prior, lh_doc_type, lh_valid_digital_sign, use_days_mean, use_days_var);
		print2DVector(raw);

	};

	cout << endl << endl;

	// Record the algorithm runtime
	std::chrono::duration<double> elapsed_sec = stop-start;
	cout << "Runtime: " << elapsed_sec.count() << endl << endl;

	// Probs normalization
	vector<double> p1(146);

	for (int i = 0; i < doc_typeTest.size(); i++) {

		// 1x2 matrix
		raw = calc_raw_prob(doc_typeTest.at(i), valid_digitalSignatureTest.at(i), use_daysTest.at(i), prior, lh_doc_type, lh_valid_digital_sign, use_days_mean, use_days_var);

		if (raw.at(0).at(0) > .5) {

			p1.at(i) = 0;

		} else if (raw.at(0).at(1) > .5) {

			p1.at(i) = 1;

		} else {};

	};

	// Confusion Matrix
	cout << "Confusion Matrix: " << endl;
	vector<vector<double>> table = confusionMatrix(p1, classificationTest);
	print2DVector(table);
	cout << endl;

	double acc = accuracy(p1, classificationTest);
	cout << "Accuracy: " << acc << endl;

	// Sensitivity = TP / (TP + FN)
	double sensitivity = (table.at(0).at(0) / (table.at(0).at(0) + table.at(1).at(0)));
	cout << "Sensitivity: " << sensitivity << endl;
	
	// Specificity = TN / (TN + FP)
	double specificity = (table.at(1).at(1) / (table.at(1).at(1) + table.at(0).at(1)));
	cout << "Specificity: " << sensitivity << endl << endl;

	return 0;

}; // Function main end





// Methods Definition

// Method to print the vector
void print2DVector(vector<vector<double>> vect) {

	for (int i = 0; i < vect.size(); i++) {

		for (int j = 0; j < vect[i].size(); j++) {

			cout << vect[i][j] << " ";

		};

		cout << endl;

	};

};

// Method to evaluate the prior probability in the train data (output variable)
vector<vector<double>> priorProb(vector<double> vect) {

	// 1x2 matrix
	vector<vector<double>> prior(1, vector<double>(2, 0));

	for (int i = 0; i < vect.size(); i++) {

		if (vect.at(i) == 0) {
			
			prior.at(0).at(0)++;
		
		} else {

			prior.at(0).at(1)++;
		
		};

	};

	prior.at(0).at(0) = prior.at(0).at(0) / vect.size();
	prior.at(0).at(1) = prior.at(0).at(1) / vect.size();

	return prior;

};

// Method to count class (input variables)
vector<vector<double>> countClass(vector<double> vect) {

	// 1x2 matrix
	vector<vector<double>> prior(1, vector<double>(2, 0));

	for (int i = 0; i < vect.size(); i++) {

		if (vect.at(i) == 0) {

			prior.at(0).at(0)++;

		} else {

			prior.at(0).at(1)++;

		};

	};

	return prior;

};

// Method to evaluate the doc type probability (likelihood)
vector<vector<double>> likelihoodDoc_type(vector<double> classification, vector<double> doc_type, vector<vector<double>> count_classification) {

	// 2x3 matrix
	vector<vector<double>> lh_doc_type = {2, vector<double> (3,0)};

	for (int i = 0; i < classification.size(); i++) {

		if (classification.at(i) == 0) {

			if (doc_type.at(i) == 1) {

				lh_doc_type.at(0).at(0)++;

			} else if (doc_type.at(i) == 2) {

				lh_doc_type.at(0).at(1)++;

			} else if (doc_type.at(i) == 3) {

				lh_doc_type.at(0).at(2)++;

			} else {};

		} else if (classification.at(i) == 1) {

			if (doc_type.at(i) == 1) {

				lh_doc_type.at(1).at(0)++;

			} else if (doc_type.at(i) == 2) {

				lh_doc_type.at(1).at(1)++;

			} else if (doc_type.at(i) == 3) {

				lh_doc_type.at(1).at(2)++;

			} else {};

		} else {};

	};

	for (int i = 0; i < lh_doc_type.size(); i++) {

		for (int j = 0; j < lh_doc_type[i].size(); j++) {

			if (i == 0) {

				lh_doc_type.at(i).at(j) == lh_doc_type.at(i).at(j) / count_classification.at(0).at(0); 

			} else if (i == 1) {
				
				lh_doc_type.at(i).at(j) = lh_doc_type.at(i).at(j) / count_classification.at(0).at(1);

			};
			
		};

	};

	return lh_doc_type;

};

vector<vector<double>> likelihoodValidDigitalSign(vector<double> classification, vector<double> valid_digitalsign, vector<vector<double>> count_valid_digitalsign) {

	// 2x2 matrix
	vector<vector<double>> lh_valid_digital_sign(2, vector<double> (2, 0));

	for (int i = 0; i < classification.size(); i++) {

		if (classification.at(i) == 0) {

			if(valid_digitalsign.at(i) == 0) {

				lh_valid_digital_sign.at(0).at(0)++;

			} else if (valid_digitalsign.at(i) == 1) {

				lh_valid_digital_sign.at(0).at(1)++;

			} else {};

		} else if (classification.at(i) == 1) {

			if (valid_digitalsign.at(i) == 0) {

				lh_valid_digital_sign.at(1).at(0)++;

			}  else if (valid_digitalsign.at(i) == 1) {

				lh_valid_digital_sign.at(1).at(1)++;

			} else {};

		} else {};

	};

	for (int i = 0; i < lh_valid_digital_sign.size(); i++) {

		for (int j = 0; j < lh_valid_digital_sign[i].size(); j++) {

			if (i == 0) {

				lh_valid_digital_sign.at(i).at(j) = lh_valid_digital_sign.at(i).at(j) / count_valid_digitalsign.at(0).at(0);

			} else if (i == 1) {

				lh_valid_digital_sign.at(i).at(j) = lh_valid_digital_sign.at(i).at(j) / count_valid_digitalsign.at(0).at(1);

			};

		};

	};

	return lh_valid_digital_sign;

};

// Method to calculate the mean of use_days train variable
vector<vector<double>> use_daysMean(vector<double> classification, vector<double> use_days, vector<vector<double>> count_use_days) {

	// 1x2 matrix
	vector<vector<double>> mean(1, vector<double>(2, 0));

	for (int i = 0; i < classification.size(); i++) {

		if (classification.at(i) == 0) {

			mean.at(0).at(0) += use_days.at(i);

		} else if (classification.at(i) == 1) {

			mean.at(0).at(1) += use_days.at(i);

		} else {};

	};

	for (int i = 0; i < mean.size(); i++) {

		for (int j = 0; j < mean[i].size(); j++) {

			if (j == 0) {

				mean.at(i).at(j) = mean.at(i).at(j) / count_use_days.at(0).at(0);

			} else if (j == 1) {

				mean.at(i).at(j) = mean.at(i).at(j) / count_use_days.at(0).at(1);

			} else {};

		};

	};

	return mean;

};


// Method to calculate the variance of use_days train variable
vector<vector<double>> use_daysVariance(vector<double> classification, vector<double> use_days, vector<vector<double>> count_use_days) {

	// 1x2 matrix
	vector<vector<double>> variance(1, vector<double> (2,0));

	// 1x2 matrix
	vector<vector<double>> mean = {use_daysMean(classification, use_days, count_use_days)};

	for (int i = 0; i < classification.size(); i++) {

		if (classification.at(i) == 0) {

			variance.at(0).at(0) += pow((use_days.at(i) - mean.at(0).at(0)), 2);

		} else if (classification.at(i) == 1) {

			variance.at(0).at(1) += pow((use_days.at(i) - mean.at(0).at(1)), 2);

		} else {};

	};

	for (int i = 0; i < variance.size(); i++) {

		for (int j = 0; j < variance[i].size(); j++) {

			if (j == 0) {

				variance.at(i).at(j) = variance.at(i).at(j) / (count_use_days.at(0).at(0) - 1);

			} else if (j == 1) {

				variance.at(i).at(j) = variance.at(i).at(j) / (count_use_days.at(0).at(1) - 1);

			} else {};

		};

	};

	return variance;

};

// Method to convert the mean and variance metrics into 2x2 matrix
vector<vector<double>> use_daysMetrics(vector<vector<double>> use_daysMean, vector<vector<double>> use_daysVariance) {

	// 2x2 matrix
	vector<vector<double>> metrics = {2, vector<double>(2,0)};

	metrics.at(0).at(0) = use_daysMean.at(0).at(0);
	metrics.at(0).at(1) = sqrt(use_daysVariance.at(0).at(0));
	metrics.at(1).at(0) = use_daysMean.at(0).at(1);
	metrics.at(1).at(1) = sqrt(use_daysVariance.at(0).at(1));

	return metrics;

};

// Evaluate the probabilities of use_days variable
double calc_use_days_lh(double v, double mean_v, double var_v) {

	double use_days_lh = 0;

	// Eval the probs
	use_days_lh = (1 / (sqrt(2 * M_PI * var_v))) * exp(-(pow((v - mean_v), 2)) / (2 * var_v));

	return use_days_lh;

};

// Deploying the Bayes Theorem
vector<vector<double>> calc_raw_prob(double doc_type, double valid_digitalsign, double use_days, vector<vector<double>> prior, vector<vector<double>> lh_doc_type, vector<vector<double>> lh_valid_digitalsign, vector<vector<double>> use_days_mean, vector<vector<double>> use_days_var) {

	// 1x2 matrix
	vector<vector<double>> raw(1, vector<double>(2,0));

	// Output variable prob
	double num_s = lh_doc_type.at(1).at(doc_type-1) * lh_valid_digitalsign.at(1).at(valid_digitalsign) * prior.at(0).at(1) * calc_use_days_lh(use_days, use_days_mean.at(0).at(1), use_days_var.at(0).at(1));

	// Input variables prob
	double num_p = lh_doc_type.at(0).at(doc_type-1) * lh_valid_digitalsign.at(0).at(valid_digitalsign) * prior.at(0).at(0) * calc_use_days_lh(use_days, use_days_mean.at(0).at(0), use_days_var.at(0).at(0));

	// Denominator
	double denominator = lh_doc_type.at(1).at(doc_type - 1) * lh_valid_digitalsign.at(1).at(valid_digitalsign) * calc_use_days_lh(use_days, use_days_mean.at(0).at(1), use_days_var.at(0).at(1)) * prior.at(0).at(1) + lh_doc_type.at(0).at(doc_type-1) * lh_valid_digitalsign.at(0).at(valid_digitalsign) * calc_use_days_lh(use_days, use_days_mean.at(0).at(0), use_days_var.at(0).at(0)) * prior.at(0).at(0);

	raw.at(0).at(1) = num_s / denominator;
	raw.at(0).at(0) = num_p / denominator;

	return raw;


};

// Confusion Matrix
vector<vector<double>> confusionMatrix(vector<double> matA, vector<double> matB) {

	// 2x2 matrix
	vector<vector<double>> table(2, vector<double>(2, 0));

	// matA = predicted
	// matB = testClass

	/*
		TP FP
		FN TN
	*/

	for (int i = 0; i < matA.size(); i++) {

		// True Negative
		if (matA.at(i) == 0 && matB.at(i) == 0) {

			table.at(1).at(1)++;

		}

		// True Positive
		else if (matA.at(i) == 1 && matB.at(i) == 1) {

			table.at(0).at(0)++;

		}

		// False Positive
		else if (matA.at(i) == 1 && matB.at(i) == 0) {

			table.at(0).at(1)++;

		}

		// False Negative
		else if (matA.at(i) == 0 && matB.at(i) == 1) {

			table.at(1).at(0)++;

		} else {};
	
	};

	return table;

};

// Accuracy
double accuracy(vector<double> matA, vector<double> matB) {

	int matARow = matA.size();
	int matBRow = matB.size();

	if (matARow != matBRow) {

		cout << "Error. The matrix dimensions should be equal." << endl;

	};

	double sum = 0;

	for (int i = 0; i < matA.size(); i++) {

		if (matA.at(i) == matB.at(i)) {

			sum++;

		};

	};

	return sum / matA.size();

};

#pragma GCC diagnostic pop
