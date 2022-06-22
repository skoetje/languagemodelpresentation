'''
pip install transformers
short example for generating summaries using the Huggingface library
'''

from transformers import BartTokenizer, BartForConditionalGeneration

# todo add a transcript to summarize
transcript = ''

# initialize bart, a sequence-to-sequence model fine tuned on cnn data (summarization)
# see:
# https://huggingface.co/facebook/bart-large-cnn
# for more info

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
inputs = tokenizer([transcript], max_length=1024, return_tensors="pt")
summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=0, max_length=80)
summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(summary)