# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils

class Seq2SeqModel(object):

	def __init__(self,
			source_vocab_size, #size of the source vocabulary.
			target_vocab_size, #size of the target vocabulary.
			buckets, #a list of pairs (I, O), e.g., [(2, 4), (8, 16)].
			size, #number of units in each layer of the model. dimension?
			num_layers, #number of layers in the model.
			max_gradient_norm, #gradients will be clipped to maximally this norm.
			batch_size, #the size of the batches used during training;
			learning_rate, #learning rate to start with.
			learning_rate_decay_factor, #decay learning rate by this much when needed.
			use_lstm=False, #if true, we use LSTM cells instead of GRU cells.
			num_samples=512, #number of samples for sampled softmax. Vocab 갯수보다 작아야 의미있다.
			forward_only=False, #if set, we do not construct the backward pass in the model.
			dtype=tf.float32): #data type

		self.source_vocab_size = source_vocab_size
		self.target_vocab_size = target_vocab_size
		self.buckets = buckets
		self.batch_size = batch_size
		self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=dtype)
		self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
		self.global_step = tf.Variable(0, trainable=False)

		# If we use sampled softmax, we need an output projection.
		# 예를 들어 [512(dim) x 40000(vocab size)]의 output projection를 사용해서 dim이 512인 vector를
		# 40000(vocab size)개인 vector로 변경할 수 있다.

		output_projection = None
		softmax_loss_function = None

		# Sampled softmax only makes sense if we sample less than vocabulary size.
		# vocab size가 너무 크면 학습하는데 많은 시간이 소요되기 때문에 sampled_loss로 dimension을 줄여주고
		# output_projection을 사용해서 작아진 dim을 다시 vocab size로 늘린다.
		if num_samples > 0 and num_samples < self.target_vocab_size:
			w_t = tf.get_variable("proj_w", [self.target_vocab_size, size], dtype=dtype) # 임의 값 생성
			w = tf.transpose(w_t)
			b = tf.get_variable("proj_b", [self.target_vocab_size], dtype=dtype) # 임의 값 생성
			output_projection = (w, b) # W: [output_size x num_decoder_symbols]  B: [num_decoder_symbols];

			def sampled_loss(labels, logits):
				labels = tf.reshape(labels, [-1, 1]) # 2차원 크기르 1로 맞추고, 나머지를 1차원으로 크기로
				# We need to compute the sampled_softmax_loss using 32bit floats to
				# avoid numerical instabilities.
				local_w_t = tf.cast(w_t, tf.float32)
				local_b = tf.cast(b, tf.float32)
				local_inputs = tf.cast(logits, tf.float32)
				return tf.cast(
				    tf.nn.sampled_softmax_loss(
				        weights=local_w_t, #[num_classes(vocabs), dim]
				        biases=local_b, #[num_classes(vocabs)]
				        labels=labels, #[batch_size, num_true]
				        inputs=local_inputs, #[batch_size, dim]
				        num_sampled=num_samples ,#The number of classes to randomly sample per batch.
				        num_classes=self.target_vocab_size),#The number of possible classes.
				    dtype)

			softmax_loss_function = sampled_loss
		
		if use_lstm:
			def single_cell():
				return tf.contrib.rnn.BasicLSTMCell(size)
		else:
			def single_cell():
				return tf.contrib.rnn.GRUCell(size)
		
		if num_layers > 1:
			cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)])
		else:
			cell = single_cell()

		# The seq2seq function: we use embedding for the input and attention.
		# output_projection : if provided and feed_previous=True, each fed previous output will first be multiplied by W and added B.
		#						(W, B) W: [output_size x num_decoder_symbols]  B: [num_decoder_symbols];
		# output_projection[0] = [hidden_size x vocab수] == [size x target_vocab_size]
		# 							== [output_size x num_decoder_symbols]
		# ->This allows to use our seq2seq models with a sampled softmax loss
		'''
		when output_projection is None, the size of the attention vectors and variables will 
		be made proportional to num_decoder_symbols, can be large.
		'''
		def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
			return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
				encoder_inputs,
				decoder_inputs,
				cell,
				num_encoder_symbols=source_vocab_size,
				num_decoder_symbols=target_vocab_size,
				embedding_size=size, #Integer, the length of the embedding vector for each symbol. (각 단어당 embedding vector)
				#output_projection 도 고려해서 학습시키기 위해서 넣어줬나? None이면 num_decoder_symbols 만큼 커지는데 너무 크다. 학습시 안좋음
				output_projection=output_projection, # ([hidden_size(size, embedding_size) x vocab수], [vocab수])
				feed_previous=do_decode,
				dtype=dtype)		

		# Feeds for inputs.
		self.encoder_inputs = []
		self.decoder_inputs = []
		self.target_weights = []

		for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
			self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                        name="encoder{0}".format(i)))
		for i in xrange(buckets[-1][1] + 1): # +1 하는 이유는 앞에 _GO 태그를 넣기 위해서
			self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                        name="decoder{0}".format(i)))
			self.target_weights.append(tf.placeholder(dtype, shape=[None],
                                        name="weight{0}".format(i)))

		# Our targets are decoder inputs shifted by one.
		# 타켓은 디코더 인풋을 왼쪽으로 한칸씩 댕긴 값이다.
		targets = [self.decoder_inputs[i + 1] for i in xrange(len(self.decoder_inputs) - 1)]

		
		# Training outputs and losses.
		# model_with_buckets : bucketing을 지원하기위한 seq2seq모델
		if forward_only:
			self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
				self.encoder_inputs, self.decoder_inputs, targets,
				self.target_weights, buckets,
				lambda x, y: seq2seq_f(x, y, True),
				softmax_loss_function=softmax_loss_function)

			# If we use output projection, we need to project outputs for decoding.
			# Sampled softmax 수 가 vocab size보다 작아서 projection(sampled softmax)가 사용된 경우
			# 결과(decoding)를 얻기 위해 을 위해 ouputs를 project할 필요가 있다.

			# batch-size by size ->  batch-size by target_vocab_size

			'''
			Then, as you can see, we construct an output projection. 
			It is a pair, consisting of a weight matrix and a bias vector. 
			If used, the rnn cell will return vectors of shape batch-size by size, 
			rather than batch-size by target_vocab_size. To recover logits, 
			we need to multiply by the weight matrix and add the biases.
			'''
			if output_projection is not None:
				for b in xrange(len(buckets)):
					self.outputs[b] = [
						# outout : [batch_size x hidden_size(vocab 수)]
					 	# output_projection[0]: [vocab수 x hidden_size].T 크기의 w
					 	#						[hidden_size x vocab수]
					 	# output_projection[1]: vacab수 만큼의 b
						tf.matmul(output, output_projection[0]) + output_projection[1]
						for output in self.outputs[b]
					]

		else:
			self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
				self.encoder_inputs, self.decoder_inputs, targets,
				self.target_weights, buckets,
				lambda x, y: seq2seq_f(x, y, False),
				softmax_loss_function=softmax_loss_function)
				#softmax_loss_function는 Sampled softmax 가 vocabulary size 보다 작을때만 사용되게
				#model을 만들 때 설정해놨다.

		# Gradients and SGD update operation for training the model.
		params = tf.trainable_variables()
		if not forward_only:
			self.gradient_norms = []
			self.updates = []
			opt = tf.train.GradientDescentOptimizer(self.learning_rate)
			for b in xrange(len(buckets)):
				gradients = tf.gradients(self.losses[b], params)
				clipped_gradients, norm = tf.clip_by_global_norm(gradients,
				                                                 max_gradient_norm)
				self.gradient_norms.append(norm)

				# apply_gradients 는 minimize 내부에서 실행할때 사용하는 
				# 값으로 여기서 minimize 으로써 사용됐다.
				self.updates.append(opt.apply_gradients(
				    zip(clipped_gradients, params), global_step=self.global_step))
		
		self.saver = tf.train.Saver(tf.global_variables())


	def step(self, session, encoder_inputs, decoder_inputs, target_weights,
				bucket_id, forward_only):

		# Check if the sizes match.
		encoder_size, decoder_size = self.buckets[bucket_id]
		if len(encoder_inputs) != encoder_size:
			raise ValueError("Encoder length must be equal to the one in bucket, %d != %d." % (len(encoder_inputs), encoder_size))
		if len(decoder_inputs) != decoder_size:
			raise ValueError("Decoder length must be equal to the one in bucket, %d != %d." % (len(decoder_inputs), decoder_size))
		if len(target_weights) != decoder_size:
			raise ValueError("Weights length must be equal to the one in bucket, %d != %d." % (len(target_weights), decoder_size))


		# Input feed: encoder inputs, decoder inputs, target_weights, as provided.
		# tensor에 innput,weight 값을 feeding 하기 위해 dic을 생성한다
		# graph는 placeholder 로 데이터를 받고 이 placeholder를 self.encoder_inputs[l]으로 가져올 수 있다.
		# 결국 이 placeholder에 넣기 위해 딕을 {self.encoder_inputs[l].name: inputs[l]} 식으로 구성한다.
		input_feed = {}
		for l in xrange(encoder_size):
			input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
		for l in xrange(decoder_size):
			input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
			input_feed[self.target_weights[l].name] = target_weights[l]

		# Since our targets are decoder inputs shifted by one, we need one more.
		# decoder_input의 마지막 값이 target으로 갔기 때문에, 0으로 채워줄 하나가 더 필요하다 
		last_target = self.decoder_inputs[decoder_size].name
		input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)


		# Output feed: depends on whether we do a backward step or not.
		if not forward_only:
			output_feed = [self.updates[bucket_id],  # Update Op that does SGD. output: outputs
							self.gradient_norms[bucket_id],  # Gradient norm. output: gradient norm
							self.losses[bucket_id]]  # Loss for this batch. output: loss
		else:
			output_feed = [self.losses[bucket_id]]  # Loss for this batch.
			for l in xrange(decoder_size):  # Output logits.
				output_feed.append(self.outputs[bucket_id][l])

		outputs = session.run(output_feed, input_feed)
		if not forward_only:
			return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
		else:
			return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.



	def get_batch(self, data, bucket_id):

		encoder_size, decoder_size = self.buckets[bucket_id]
		encoder_inputs, decoder_inputs = [], []

		# Get a random batch of encoder and decoder inputs from data,
		# pad them if needed, reverse encoder inputs and add GO to decoder.
		# batch_size만큼 해당 bucket에서 데이터(input_seq, output_seq)를 가져와서
		# 'PAD_ID'와 'GO_ID'를 추가하고 encoder_inputs, decoder_inputs에 넣는다.
		# input은 넣기전에 inverse시킨다 (성능이 더 좋아짐) 
		for _ in xrange(self.batch_size):
			encoder_input, decoder_input = random.choice(data[bucket_id])

			# Encoder inputs are padded and then reversed.
			encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
			encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

			# Decoder inputs get an extra "GO" symbol, and are padded then.
			decoder_pad_size = decoder_size - len(decoder_input) - 1
			decoder_inputs.append([data_utils.GO_ID] + decoder_input +
            						[data_utils.PAD_ID] * decoder_pad_size)

		# Now we create batch-major vectors from the data selected above.
		batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

		# Batch encoder inputs are just re-indexed encoder_inputs.
		# 입력으로 사용하기 위해 encoder_inputs를 transport 함
		# np.array(incoder_inputs).T 한거하고 같은듯
		for length_idx in xrange(encoder_size):
			batch_encoder_inputs.append(
			np.array([encoder_inputs[batch_idx][length_idx]
				for batch_idx in xrange(self.batch_size)], dtype=np.int32))

		# Batch decoder inputs are re-indexed decoder_inputs, we create weights.
		for length_idx in xrange(decoder_size):
			batch_decoder_inputs.append(
			np.array([decoder_inputs[batch_idx][length_idx]
				for batch_idx in xrange(self.batch_size)], dtype=np.int32))

			# Create target_weights to be 0 for targets that are padding.
			# padding인 값의 w 는 0으로 준다.
			batch_weight = np.ones(self.batch_size, dtype=np.float32)
			for batch_idx in xrange(self.batch_size):
				# We set weight to 0 if the corresponding target is a PAD symbol.
				# The corresponding target is decoder_input shifted by 1 forward.
				if length_idx < decoder_size - 1:
					target = decoder_inputs[batch_idx][length_idx + 1]
				if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
					batch_weight[batch_idx] = 0.0
			
			batch_weights.append(batch_weight)

		return batch_encoder_inputs, batch_decoder_inputs, batch_weights


	def restore_last_session(self, ckpt_path):
		# create a session
		sess = tf.Session()
		# get checkpoint state
		ckpt = tf.train.get_checkpoint_state(ckpt_path)
		# restore session
		if ckpt and ckpt.model_checkpoint_path:
			self.saver.restore(sess, ckpt.model_checkpoint_path)
			return sess
		else:
			return None

