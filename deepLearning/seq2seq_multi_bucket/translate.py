# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import math
import os
import random
import sys
import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
import seq2seq_model_v2

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.(hidden_size, output_dimension)")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("from_vocab_size", 40000, "English vocabulary size.")
tf.app.flags.DEFINE_integer("to_vocab_size", 40000, "French vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_string("from_train_data", None, "Training data.")
tf.app.flags.DEFINE_string("to_train_data", None, "Training data.")
tf.app.flags.DEFINE_string("from_dev_data", None, "Training data.")
tf.app.flags.DEFINE_string("to_dev_data", None, "Training data.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0, "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200, "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False, "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", True, "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False, "Train using fp16 instead of fp32.")

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

def read_data(source_path, target_path, max_size=None):
	"""Read data from source and target files and put into buckets.

	Read data from source and target files and put into buckets.
	Args:
		source_path: path to the files with token-ids for the source language.
		target_path: path to the file with token-ids for the target language;
			it must be aligned with the source file: n-th line contains the desired
			output for n-th line from the source_path.
			
			input과 output은 같은 라인에 매칭돼야한다.
			input과 output은 embedding된 id(index) 값이다.

		max_size: maximum number of lines to read, all other will be ignored;
			if 0 or None, data files will be read completely (no limit).
	Returns:
		data_set: a list of length len(_buckets); data_set[n] contains a list of
			(source, target) pairs read from the provided data files that fit
			into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
			len(target) < _buckets[n][1]; source and target are lists of token-ids.

			버킷인덱스에 리스트들이 모여있고. 그 리스트들은 (input output) 튜플들로 모여있다.

	"""

	data_set = [[] for _ in _buckets]
	with tf.gfile.GFile(source_path, mode="r") as source_file:
		with tf.gfile.GFile(target_path, mode="r") as target_file:
			source, target = source_file.readline(), target_file.readline()
			counter = 0
			while source and target and (not max_size or counter < max_size):
				counter += 1
				if counter % 100000 == 0:
					print("  reading data line %d" % counter)
					sys.stdout.flush()
				source_ids = [int(x) for x in source.split()]
				target_ids = [int(x) for x in target.split()]
				target_ids.append(data_utils.EOS_ID)
				for bucket_id, (source_size, target_size) in enumerate(_buckets):
					if len(source_ids) < source_size and len(target_ids) < target_size:
						data_set[bucket_id].append([source_ids, target_ids])
						break
				source, target = source_file.readline(), target_file.readline()
		
		return data_set

def create_model(session, forward_only):
	"""Create translation model and initialize or load parameters in session."""
	dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
	model = seq2seq_model_v2.Seq2SeqModel(
		FLAGS.from_vocab_size,
		FLAGS.to_vocab_size,
		_buckets,
		FLAGS.size,
		FLAGS.num_layers,
		FLAGS.max_gradient_norm,
		FLAGS.batch_size,
		FLAGS.learning_rate,
		FLAGS.learning_rate_decay_factor,
		forward_only=forward_only,
		dtype=dtype)
  
	ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
	if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
		print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
		model.saver.restore(session, ckpt.model_checkpoint_path)
	else:
		print("Created model with fresh parameters.")
		session.run(tf.global_variables_initializer())

	return model


# //사전준비
# 학습데이터를 다운 -> 압출 풀기 -> 
# 특정 경로에 vocab, tokenized train set 을 파일로 저장
# 특정 경로를 변수에 저장

# //세션시작
# 모델 생성 -> 파일 경로로부터 dev_set, train_set 을 로드.

# //Train 시작: epoch는 현재 무한 루프 인 듯
# bucket_id를 랜덤으로 선택 -> bucket_id에 맞는 train에 필요한 값들을 batch_size만큼 로그
# 로드된 encoder_inputs, decoder_inputs, target_weights를 가지고 train
# target은 decoder_input을 왼쪽으로 한칸씩 시프트한 값
# check point 마자 train 값을 저장
def train():
	"""Train a en->fr translation model using WMT data."""
	from_train = None
	to_train = None
	from_dev = None
	to_dev = None

	# 다운로드 후에 압출을 풀어놓은 데이터가 있으면 사용하고 아니면 다운 받아서 사용한다.
	# 그 후에 prepare_data, prepare_wmt_data 모두 내부에서는 만들어놓은 vocab, tokenize 
	# 데이터가 있으면 그대로 반환하고, 없으면 새로 생성해서 반환한다.
	if FLAGS.from_train_data and FLAGS.to_train_data:
		from_train_data = FLAGS.from_train_data
		to_train_data = FLAGS.to_train_data
		from_dev_data = from_train_data
		to_dev_data = to_train_data

		# 지정된 dev 경로가 따로 없으면 train 경로를 대입한다.
		if FLAGS.from_dev_data and FLAGS.to_dev_data:
			from_dev_data = FLAGS.from_dev_data
			to_dev_data = FLAGS.to_dev_data

		from_train, to_train, from_dev, to_dev, _, _ = data_utils.prepare_data(
			FLAGS.data_dir,
			from_train_data,
			to_train_data,
			from_dev_data,
			to_dev_data,
			FLAGS.from_vocab_size,
			FLAGS.to_vocab_size)
	else:
		# Prepare WMT data.
		print("Preparing WMT data in %s" % FLAGS.data_dir)
		from_train, to_train, from_dev, to_dev, _, _ = data_utils.prepare_wmt_data(
			FLAGS.data_dir, FLAGS.from_vocab_size, FLAGS.to_vocab_size)

	with tf.Session() as sess:
		# Create model.
		print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
		model = create_model(sess, False)

		# Read data into buckets and compute their sizes.
		print ("Reading development and training data (limit: %d)." % FLAGS.max_train_data_size)
		dev_set = read_data(from_dev, to_dev) #모두 읽어온다.
		train_set = read_data(from_train, to_train, FLAGS.max_train_data_size) #max값 만큼 일어온다.
		train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))] #bucket당 train size를 가져온다.
		train_total_size = float(sum(train_bucket_sizes)) #모든 bucket size를(전체 train (in, out)수) 가져온다.

		# A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
		# to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
		# the size if i-th training bucket, as used later.
		# 전체 bucket size 중에서 n번째 bucket size까지 합의 비율
		# if train_bucket_sizes: [14, 32, 11] -> train_buckets_scale: [0.24561403508771928, 0.8070175438596491, 1.0]
		train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size 
								for i in xrange(len(train_bucket_sizes))]

		# This is the training loop.
		step_time, loss = 0.0, 0.0
		current_step = 0
		previous_losses = []

		while True:
			# Choose a bucket according to data distribution. We pick a random number
			# in [0, 1] and use the corresponding interval in train_buckets_scale.
			# 랜덤으로 0, 1 사이에서 하나를 뽑아서 분포도에 맞는 bucket_id를 가져온다
			# -> 랜덤으로 분포도에 맞게 bucket_id를 가져올 수 있다.
			random_number_01 = np.random.random_sample()
			bucket_id = min([i for i in xrange(len(train_buckets_scale))
								if train_buckets_scale[i] > random_number_01])


			# Get a batch and make a step.
			start_time = time.time()
			# train_set에서 bucket_id에 해당하는 데이터를(in, out) transport해서 batch_size만큼 가져온다. 
			encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id)

			# 해당 bucket_id로 batch_size만큼 train한다. <- one step
			_, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
											target_weights, bucket_id, False)
			step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
			loss += step_loss / FLAGS.steps_per_checkpoint
			current_step += 1

			# Once in a while, we save checkpoint, print statistics, and run evals.
			# check point마다 저장하고 통계를 출력한다.
			if current_step % FLAGS.steps_per_checkpoint == 0:
				# Print statistics for the previous epoch.
				perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
				print ("global step %d learning rate %.4f step-time %.2f perplexity "
							"%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
							step_time, perplexity))

				# Decrease learning rate if no improvement was seen over last 3 times.
				# 전전보다 loss가 줄지 않으면 learning rate를 줄인다.
				if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
					sess.run(model.learning_rate_decay_op)
				previous_losses.append(loss)

				# Save checkpoint and zero timer and loss.
				checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
				model.saver.save(sess, checkpoint_path, global_step=model.global_step)
				step_time, loss = 0.0, 0.0

				# Run evals on development set and print their perplexity.
				for bucket_id in xrange(len(_buckets)):
					if len(dev_set[bucket_id]) == 0:
						print("  eval: empty bucket %d" % (bucket_id))
						continue
					encoder_inputs, decoder_inputs, target_weights = model.get_batch(dev_set, bucket_id)
					_, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
					eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float("inf")
					print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
				
				sys.stdout.flush()

def decode():
	with tf.Session() as sess:

		# Create model and load parameters.
		model = create_model(sess, True)
		model.batch_size = 1  # We decode one sentence at a time.

		# Load vocabularies.
		en_vocab_path = os.path.join(FLAGS.data_dir, "vocab%d.from" % FLAGS.from_vocab_size)
		fr_vocab_path = os.path.join(FLAGS.data_dir, "vocab%d.to" % FLAGS.to_vocab_size)

		# ouput: vocab(word2idx), reverse_vocab(idx2word)
		en_vocab, _ = data_utils.initialize_vocabulary(en_vocab_path)
		_, rev_fr_vocab = data_utils.initialize_vocabulary(fr_vocab_path)

		# Decode from standard input.
		sys.stdout.write("> ")
		sys.stdout.flush()
		sentence = sys.stdin.readline()
		while sentence:
			# Get token-ids for the input sentence.
			token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), en_vocab)

			# Which bucket does it belong to?
			# 문장에 맞는 bucket_id를 찾는다.
			bucket_id = len(_buckets) - 1
			for i, bucket in enumerate(_buckets):
				if bucket[0] >= len(token_ids):
					bucket_id = i
					break
			else:
				logging.warning("Sentence truncated: %s", sentence)

			# Get a 1-element batch to feed the sentence to the model.
			# bucket_id에 해당하는 하나의 encoder_input만 bach_size만큼 가져온다.
			encoder_inputs, decoder_inputs, target_weights = model.get_batch( 
				{bucket_id: [(token_ids, [])]}, bucket_id)

			# Get output logits for the sentence.
			# 결과를 가져온다.
			_, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
												target_weights, bucket_id, True)


			# greedy decoding: 우선 다음 단어 하나를 최고의 확률값을 가지는 것으로 예측. 이 단어를 통해 다음 최고 활률값 예측 
			# beam search: 한 단계에서 단어를 n개 예측후 조합. 예로 한단계에서 3개의 단어를 예측, 각각의 단어를 이용 다음 단어를 3개 예측. 
			# 그런다음 각각의 조합을 묶어서 조합의 가능성이 가장 높은 것을 선택 (대부분 beam size는 3~10)

			# This is a greedy decoder - outputs are just argmaxes of output_logits.
			# argmax로 가장 높은 확률이였던 demension의 index로 outputs를 만든다.
			outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits] # batch_size가 한개일때만
			# outputs = [np.argmax(logit, axis=1) for logit in output_logits]

			# If there is an EOS symbol in outputs, cut them at that point.
			# EOS_ID 전까지 가져온다.
			if data_utils.EOS_ID in outputs:
				outputs = outputs[:outputs.index(data_utils.EOS_ID)]

			# Print out French sentence corresponding to outputs.
			# ouput을 다시 sentence(words)로 변경한다.
			print(" ".join([tf.compat.as_str(rev_fr_vocab[output]) for output in outputs]))
			print("> ", end="")

			sys.stdout.flush()
			sentence = sys.stdin.readline()

# 샘플을 가지고 학습을 테스트한다.
def self_test():
	"""Test the translation model."""
	print('Test the translation model')
	with tf.Session() as sess:
		print("Self-test for neural translation model.")
		# Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
		model = seq2seq_model_v2.Seq2SeqModel(10, 10, [(3, 3), (6, 6)], 32, 2,
												5.0, 16, 0.3, 0.99, num_samples=8)
		
		# 문장 하나에 대한 output을 얻기 위해서
		model.batch_size = 1

		# 변수 초기화
		sess.run(tf.global_variables_initializer())

		# Fake data set for both the (3, 3) and (6, 6) bucket.
		data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])], #bucket (3, 3) 의 data
					[([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])]) #bucket (6, 6)의 data

		for _ in xrange(5):  # Train the fake model for 5 steps.
			bucket_id = random.choice([0, 1])
			encoder_inputs, decoder_inputs, target_weights = model.get_batch(data_set, bucket_id)
			model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)


		#샘플 input(token_ids==문장)으로 결과를 가져오기
		bucket_id = 0 #임의의 문자열에 대한 bucket id (buckets = [(3, 3), (6, 6)])
		token_ids = [1, 2, 1] #임의의 문자열 idx
		encoder_inputs, decoder_inputs, target_weights = model.get_batch( 
				{bucket_id: [(token_ids, [])]}, bucket_id)
		_, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
												target_weights, bucket_id, True)

		# ouput logits shape : [bucket_length, batch_size, embedding_size]
		print(len(output_logits)) # 3(output 길이 : bucket의 아웃풋 최대 값)
		print(len(output_logits[0])) # 1 (batch_size: incoder_input수만큼 결과를 가져온다) 
		print(len(output_logits[0][0])) # 32 hidden_size(embedding_size로 해야 값을 얻을 수 있을듯)

		outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
		print(outputs)

		# 문장에 EOS_ID 태그가 있으면 그 전까지 잘라낸다.
		if data_utils.EOS_ID in outputs:
			outputs = outputs[:outputs.index(data_utils.EOS_ID)]

def main(_):
	if FLAGS.self_test:
		self_test()
	elif FLAGS.decode:
		decode()
	else:
		train()

if __name__ == "__main__":
	tf.app.run()