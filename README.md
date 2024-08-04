# gpt2-peft
PyTorch implementation and reproduction of the OpenAI GPT-2 architecture + PEFT methods such as LoRA/QLoRA/DoRA/QDoRA (from scratch).

Instruction finetuning is primarily accomplished through LoRA and DoRA, a Parameter Efficient Finetuning (PEFT) method implemented from scratch. Supervised Finetuning (SFT) and/or continued pretraining is also done for experimentation purposes. 

## Task List
- [x] GPT-2 architecture implementation from PyTorch
- [x] LoRA adapter implementation
- [x] DoRA adapter implementation
- [x] QLoRA implementation (from bnb)
- [x] QDoRA implementation (from bnb)
- [ ] [BUG] Quantized model loading
- [ ] IA3
- [ ] Instruction benchmarking

## Future Work
Future work include implementation of extensions to other PEFT methods such as IA3, a drop-in interface with other potentially stronger models, as well as prompt engineering techniques such as RAG and usage of a vector database. 
Before these, I intend to implement proper benchmarking/evaluation to quantify the improvement from finetuning. 

## Results
Current finetuned results using LoRA, QLoRA, and DoRA on gpt2-large and gpt2-medium base models trained on the Alpaca-52k and LIMA (1k) datasets show promising results. All training and inference done on a single RTX3080 10GB GPU. Quantization is performed using the [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) library. 

## Example outputs BEFORE LoRA finetuning:
```
---PROMPT---
Provide a numbered list of the 10 largest tech companies.
---OUTPUT---


How much is the tax paid by the technology industry? In the mid-2000s, the largest 1 percent of companies actually paid 14 percent of the corporate income tax; the top 0.01 percent made more than $1.6 trillion from federal taxes in 2014, according to the Tax Policy Center. The bottom 90 percent of companies paid zero corporate income tax, because their taxes were so minimal.

Advertisement


Let's look at the 10 biggest tech companies by industry.

Net worth

I'm really not a good judge of value, but since these are the types of lists you're reading, I'd guess that this will be about $12,000,000. This is a big number for companies in industries like health care, which is why the number of tech firms in health care is so astounding.
```
```
---PROMPT---
Provide a numbered list of the 10 largest tech companies.
---OUTPUT---


The companies could say for whom they provide products or services. Employees at each company could say for whom they work.

The participants in the survey would then be asked: "When in the year 2014, how many of your work colleagues were, or had they been, in the following roles in tech or other fields: Product manager, Developer, QA, Sales, Marketing ...".

The respondents could then say: "What is the percentage of the work of your colleagues and friends who fit into the first category?"
```
```
---PROMPT---
Provide a numbered list of the 10 largest tech companies.
---OUTPUT---


A few reasons for wanting to change your company's name.

1. It reduces tax headaches.

It's a very big business in the United States, and the US tax code can be a huge hassle. You may have wondered what happens to your corporate profits if you end up changing your name.

The US Treasury Department will require that a change be made, and even if you end up changing it, you still won't receive any tax benefits.

2. It prevents confusion.

The name change can change a person's identity completely. With the changing of the company name, it'll only confuse people even more.

A name change will also have the potential to affect your employees too.

In fact, it's been reported that companies hire people to change their name as well, as it allows them to increase their profits.

3. It's for your employees.

In case you or your employees are planning on the corporate name change, it's usually worth the investment to consider it.

As a result of an employee's name change, it would give employees the opportunity to have a professional address
```

### Example outputs AFTER LoRA finetuning:
```
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
```
```
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
```
```
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
```

Not perfect, but significantly qualitatively better. On a RTX3080, finetuning with LoRA rank=8, alpha=16, takes ~40 minutes.
