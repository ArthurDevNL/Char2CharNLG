For this exercise, we built a char2char model that generates a sequence of characters based on an feature vector. The feature vector is basically a representation of the meaning representation where there is a 1 if a given type/value pair is present or a 0 otherwise. We created a total of 35 features for this representation. For the character generation we decided to leave out uppercase letters and padded/clipped the sentences to a maximum length of 150.

To run:
	python learn_model.py –-train_dataset <pathname_to_train_dataset> –-output_model_file <pathname_to_model_file> 
	python test_model.py –-test_dataset <pathname_to_test_dataset> –-ouput_test_file <pathname_to_results_testfile> --input_model_file <pathname_to_model_file>