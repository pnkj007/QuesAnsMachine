{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Toshi\\Anaconda3\\envs\\R_env\\lib\\site-packages\\torch\\storage.py:34: FutureWarning: pickle support for Storage will be removed in 1.5. Use `torch.save` instead\n",
      "  warnings.warn(\"pickle support for Storage will be removed in 1.5. Use `torch.save` instead\", FutureWarning)\n"
     ]
    }
   ],
   "source": [
    "from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering\n",
    "import torch\n",
    "import pickle\n",
    "\n",
    "class QuestionAnsweringModel:\n",
    "    def __init__(self):\n",
    "        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased',return_token_type_ids = True)\n",
    "        self.model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')\n",
    "    \n",
    "    def encode(self,question,context):\n",
    "        encoded = self.tokenizer.encode_plus(question, context)\n",
    "        return encoded[\"input_ids\"], encoded[\"attention_mask\"]\n",
    "\n",
    "    def decode(self,token):\n",
    "        answer_tokens = self.tokenizer.convert_ids_to_tokens(token , skip_special_tokens=True)\n",
    "        return self.tokenizer.convert_tokens_to_string(answer_tokens)\n",
    "\n",
    "    def predict(self,question,context):\n",
    "        input_ids, attention_mask = self.encode(question,context)\n",
    "        start_scores, end_scores = self.model(torch.tensor([input_ids]), attention_mask=torch.tensor([attention_mask]))\n",
    "        ans_tokens = input_ids[torch.argmax(start_scores) : torch.argmax(end_scores)+1]\n",
    "        answer = self.decode(ans_tokens)\n",
    "        return answer\n",
    "\n",
    "pickle.dump(QuestionAnsweringModel(), open('model.pkl','wb'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
