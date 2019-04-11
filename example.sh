DATA_DIR=/mounts/work/philipp/tmp/data/
TMP_DIR=/mounts/work/philipp/tmp/tmp/
RESULT_DIR=/mounts/work/philipp/tmp/results/
mkdir $DATA_DIR
mkdir $RESULT_DIR
mkdir $TMP_DIR

# Get data
python utils/download_data.py --data_dir $DATA_DIR


# Run Lexicon Induction

python -m lexind.prepare.py \
		--embeddings ${DATA_DIR}cc.de.300.vec \
		--lex_train ${DATA_DIR}lexicon_train.txt \
		--lex_test ${DATA_DIR}lexicon_test.txt \
		--store ${TMP_DIR}lexind

for method in densray,binary densray,continuous regression,svm regression,svr regression,linear regression,logistic
do
	method=densray,binary
	python -m lexind.run.py \
		--embeddings ${TMP_DIR}lexind,embeddings.txt \
		--lex_train ${DATA_DIR}lexicon_train.txt \
		--lex_train_version countable \
		--lex_test ${DATA_DIR}lexicon_test.txt \
		--lex_test_version countable \
		--store ${TMP_DIR}lexind \
		--densray__weights 0.5,0.5 \
		--method $method

	python -m lexind.evaluate.py \
		--lex_true ${DATA_DIR}lexicon_test.txt \
		--lex_true_version countable \
		--lex_pred ${TMP_DIR}lexind,$method.predictions \
		--lex_pred_version continuous \
		--store ${RESULT_DIR}lexind \
		--method $method
done

# Run Word Analogy Task
python -m analogy.solve_analogy_task.py \
--embeddings ${DATA_DIR}cc.en.300.vec \
--bats false \
--analogies ${DATA_DIR}questions-words.txt \
--load_first_n 100000 \
--method regression,svm \
--pred_method clcomp \
--use_proba false \
--store ${RESULT_DIR}ga,regression,svm




