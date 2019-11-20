To train the model:
bash ngram_lm_train.bash mle 1 UD_English-EWT/en_ewt-ud-train.conllu UD_English-EWT/en_ewt-ud-dev.conllu 1gram.lm 1

To test the model:
bash ngram_lm_eval.bash 1gram.lm UD_English-EWT/en_ewt-ud-test.conllu 