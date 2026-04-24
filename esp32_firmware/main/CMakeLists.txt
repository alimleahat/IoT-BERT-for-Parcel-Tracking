idf_component_register(SRCS "main.c" "tokenizer.c" "entity.c" "inference.cc" "custom_gelu.cc" "custom_fc.cc"
                       INCLUDE_DIRS "."
                       EMBED_FILES "bert_model.tflite"
                       EMBED_TXTFILES "vocab.txt")
