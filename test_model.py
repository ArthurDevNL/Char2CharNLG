from keras.models import Model, model_from_json
from keras.layers import Input, LSTM, Dense
import pandas as pd
import numpy as np
import json
import argparse
import pickle
import preprocessing

def main(args):
    num_encoder_tokens = 33
    num_decoder_tokens = 57
    latent_dim = 256

    test_filepath = args.test_dataset[0]
    df_test = pd.read_csv(test_filepath, encoding='utf8')

    model_path = args.model_file[0] + '/'
    def load_model(name):
        json_file = open(model_path + name + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights(model_path + name + ".h5")
        return model

    encoder_model = load_model('encoder')
    decoder_model = load_model('decoder')

    char2id = json.load(open(model_path + 'char2id.json'))
    id2char = {v:k for k,v in char2id.items()}

    max_decoder_seq_length = 150
    def decode_sequence(input_seq):
        # Encode the input as state vectors.
        states_value = encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, num_decoder_tokens))

        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, char2id['<bos>']] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = id2char[sampled_token_index]
            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '<eos>' or len(decoded_sentence) > max_decoder_seq_length):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        return decoded_sentence

    type2id = pickle.load(open(model_path + "type2id.json", "rb" ))

    # ************* Feature preprocessing *************
    # mrs = meaning representations
    mrs = df_test.MR.values

    structered_mrs = [preprocessing.structure_mr_string(s) for s in mrs]
    X_feature_vectors = np.array([preprocessing.to_feature_vector(x, type2id) for x in structered_mrs])

    # Produce output data
    with open(args.output_test_file[0], 'w') as f:
        for i in range(len(X_feature_vectors)):
            mr = structered_mrs[i]
            decoded = decode_sequence(X_feature_vectors[i].reshape((1,1,len(type2id))))
            atts = dict(mr)

            decoded = decoded.replace('<name>', atts['name'])
            if 'near' in atts:
                decoded = decoded.replace('<near>', atts['near'])
            if '<eos>' in decoded:
                decoded = decoded.replace('<eos>','')

            f.write("%s\n" % decoded)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dataset", dest='test_dataset', nargs=1, help="Path to the test dataset")
    parser.add_argument("--output_test_file", dest='output_test_file', nargs=1, help="Path to output the results to")
    parser.add_argument("--input_model_file", dest='model_file', nargs=1, help="Path to the trained model")
    args = parser.parse_args()

    main(args)