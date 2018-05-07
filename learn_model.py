from keras.models import Model
from keras.layers import Input, LSTM, Dense
import json
import pandas as pd
import numpy as np
import pickle
import argparse
import os
import preprocessing

def main(args):
    df = pd.read_csv(args.train_dataset[0], encoding='utf8')

    # ************* Feature preprocessing *************
    # mrs = meaning representations
    mrs = df.mr.values

    type2id = preprocessing.construct_type_mapping(mrs)
    structered_mrs = [preprocessing.structure_mr_string(s) for s in mrs]
    X_feature_vectors = np.array([preprocessing.to_feature_vector(x, type2id) for x in structered_mrs])


    # ************* Output preprocessing *************
    # Replace the name and near values from the meaning representation with a specific token
    refs = df.ref.values

    # Setup the vocabulary and construct the char2id mapping
    vocab = preprocessing.construct_vocab(refs, structered_mrs)
    char2id = {vocab[i]:i for i in range(len(vocab))}
    converted_sents = [preprocessing.convert_ref(r, char2id) for r in refs]

    # Convert the list of integers to a one hot vector matrix
    max_seq_len = 150
    X_data = np.array([preprocessing.one_hot_ref(r, max_seq_len, vocab) for r in converted_sents])


    # ************* Modeling *************
    # For the modeling part, we used the model proposed on https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
    #  and transformed it to fit the data of this exercise.
    encoder_input_data = X_feature_vectors.reshape((len(refs), 1, len(type2id)))
    decoder_input_data = X_data

    # Shift the target data and pad it 
    npad = ((0, 0), (0, 1), (0, 0))
    decoder_target_data = np.pad(X_data[:,1:,:], pad_width=npad, mode='constant', constant_values=0)

    batch_size = 64
    epochs = 10 
    latent_dim = 256
    num_samples = df.shape[0]

    num_encoder_tokens = len(type2id)
    num_decoder_tokens = len(vocab)

    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = LSTM(latent_dim, return_state=True)
    _, state_h, state_c = encoder(encoder_inputs)

    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, num_decoder_tokens))

    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the 
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Run training
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=batch_size, epochs=epochs)

    # Construct the model used for inference
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    model_dir = args.output_model_file[0] + '/' 


    # Saves the given model to the model directory
    def save_model(model, name):
        model_json = model.to_json()
        with open(model_dir + name + ".json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights(model_dir + name + ".h5")

    # Save the encoder/decoder inference model
    save_model(encoder_model, 'encoder')
    save_model(decoder_model, 'decoder')

    # Save the vocabulary
    with open(model_dir + 'char2id.json', 'w') as outfile:
        json.dump(char2id, outfile)

    # Save the type mapping
    pickle.dump(type2id, open(model_dir + 'type2id.json', 'wb'))    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset", dest='train_dataset', nargs=1, help="Path to the train dataset")
    parser.add_argument("--output_model_file", dest='output_model_file', nargs=1, help="Path to output the model to")
    args = parser.parse_args()

    model_dir = args.output_model_file[0]
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    main(args)