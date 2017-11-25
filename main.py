import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import keras
from sklearn.metrics import log_loss
from keras.layers import Embedding, Input, Lambda, LSTM, Dense, Dropout
from keras.models import Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import load_data as ld
import argparse

def euclidean_distance(vects):
	x, y = vects
	return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def eucl_dist_output_shape(shapes):
	shape1, shape2 = shapes
	return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
	'''
	contrastive loss function
	input : true label and predicted label
	output : value of contrastive loss	
	'''
	margin=1
    	return (K.mean((1 - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0))))


def model(max_len, vocab, args):
	'''
	model definition
	input : max length of embeddings, vocabulary, arguments
	output : compiled model
	'''
	embedding_dim	= 128
	num_units 	= 128
	learning_rate	= args.lr

	sequence_input_1	=   Input(shape=(max_len,), dtype='int32')
	sequence_input_2	=   Input(shape=(max_len,), dtype='int32')
	q_embedding_layer       =   Embedding(len(vocab)+1, embedding_dim, input_length=max_len, trainable=True)
	q_bilstm_layer_1        =   LSTM(num_units, return_sequences=True, input_shape=(max_len, embedding_dim))
	dropout1		=   Dropout(0.2)
	q_bilstm_layer_2        =   LSTM(num_units, return_sequences=False, input_shape=(max_len, embedding_dim))
	q_TDD_layer        	=   Dense(num_units, activation='relu')

	out1                	=   q_TDD_layer(q_bilstm_layer_2(dropout1((q_bilstm_layer_1(q_embedding_layer(sequence_input_1))))))
	out2                	=   q_TDD_layer(q_bilstm_layer_2(dropout1((q_bilstm_layer_1(q_embedding_layer(sequence_input_2))))))

	out_final       	=   Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([out1, out2])
	model        	       	=   Model(input=[sequence_input_1, sequence_input_2], output=out_final)

	opt = keras.optimizers.Adam(lr=learning_rate, clipvalue=args.clipvalue, clipnorm=args.clipnorm)
	model.compile(optimizer=opt, loss=contrastive_loss, metrics=['acc'])
	
	return model

def train(model, data, args):
	'''
	train the model
	input : (model, data, arguments)
	output : Trained model
	'''
	nq1_train, nq2_train, label_train = data

	checkpoint            =   ModelCheckpoint('model_weights.hdf5', monitor='val_loss', verbose=0, save_best_only=True, mode='min')
	callbacks_list          =   [checkpoint]

	print 'Training...'
	model.fit([nq1_train[1000:], nq2_train[1000:]], label_train[1000:], nb_epoch=args.epochs, batch_size=args.batch_size, validation_data=([nq1_train[:1000], nq2_train[:1000]], label_train[:1000]),shuffle=True,callbacks	=callbacks_list)


def test(model, data):
	'''
	test the model
	input : model, data
	output : log loss value
	'''
	nq1_test, nq2_test, label_test = data
	pred_labels = model.predict([nq1_test, nq2_test])
	logloss = log_loss(label_test, pred_labels)
	return logloss



if __name__ == "__main__":

	# setting the hyper parameters
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', default=32, type=int)
	parser.add_argument('--epochs', default=50, type=int)
	parser.add_argument('--is_training', default=1, type=int)
	parser.add_argument('--data_path', default='data/')
	parser.add_argument('--lr', default=0.001, type=float)
	parser.add_argument('--clipvalue', default=8.0, type=float)
	parser.add_argument('--clipnorm', default=10.0, type=float)
	args = parser.parse_args()
	
	#load data
	nq1_train, nq2_train, nq1_test, nq2_test, label_train,label_test, max_len, vocab = ld.load()

	input_dim, input_length, output_dim = 244, 55, 198
	#define model
	model = model(max_len, vocab, args)
	
	# train or test
	if args.is_training:
		hist = train(model=model, data=(nq1_train, nq2_train, label_train), args=args)
	else:  # load weights for testing
		model.load_weights('model_weights.hdf5')
		logloss = test(model=model, data=(nq1_test, nq2_test, label_test))
		print(" LOG LOSS with best model : ", logloss)
