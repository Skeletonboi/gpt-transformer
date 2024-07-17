# gpt2-plus
PyTorch implementation and reproduction of the OpenAI GPT-2 architecture + LoRA finetuning (all from scratch).

Instruction finetuning is done using LoRA, a Parameter Efficient Finetuning (PEFT) method implemented from scratch. Supervised Finetuning (SFT) and/or continued pretraining is also done for experimentation purposes. 

## Future Work
Future work include implementation of extensions to other PEFT methods such as QLoRA and IA3, as well as prompt engineering techniques such as RAG.

## Results
Current finetuned results using LoRA on gpt2-large base model trained on the Alpaca-52k and LIMA (1k) show promising results.

Example output:

'''
---PROMPT---
Provide a numbered list of the 10 largest tech companies.
---OUTPUT---
The 10 largest technology companies in the world are:

1. Apple (AAPL) with a market cap of $200 billion
2. Microsoft (MSFT) with a market capitalisation of $90 billion
3. Google (GOOG) with a market capitalisation of $90 billion
4. Amazon (AMZN) with a market cap of $85 billion
5. IBM (IBM) with a market cap of $65 billion
6. Cisco (CSCO) with a market cap of $60 billion
7. Amazon Web Services (AWS) with a market cap of $45 billion
8. Tesla (TSLA) with a market cap of $27 billion
9. Alibaba (BABA) with a market cap of $21 billion
10. Facebook (FB) with a market cap of $4.1 billion.
---PROMPT---
Provide a numbered list of the 10 largest tech companies.
---OUTPUT---
Based on a recent report by Reuters, some of the largest technology companies are:

1. Apple
2. Google and Facebook
3. Microsoft
4. Facebook and Twitter 
5. Amazon
6. Adobe and Salesforce
7. Apple
8. Adobe and Oracle
---PROMPT---
Provide a numbered list of the 10 largest tech companies.
---OUTPUT---
As a AI model, I don't access or access data from the Internet in the way that a human would, so I can't provide a specific list of the top 10 largest tech companies. Some common names or terms associated with tech companies can include:

1. Amazon (a multi-billion-selling company)
2. eBay (a multi-billion-selling internet company)
3. Google (a multi-billion-selling company)
4. Apple (a multi-billion-selling internet company)
5. Microsoft (a multi-billion-selling internet company) 
6. Facebook (a multi-billion-selling internet company)
7. Amazon (a multi-billion-selling internet company)
8. Netflix (a multi-billion-selling internet company)
9. Facebook (a multi-billioning internet company)
10. Google (a multi-billion-ing internet company)
'''