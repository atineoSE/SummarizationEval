python LLM.py --model mixtral_8x7b_instruct --file ./transcripts/awesome_nature_100.csv --batch_size 32 &> ./out/mixtral_32_100.out
python LLM.py --model llama2_70b_chat --file ./transcripts/awesome_nature_100.csv --batch_size 16 &> ./out/llama2_16_100.out
