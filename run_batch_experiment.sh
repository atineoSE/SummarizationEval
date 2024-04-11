python LLM.py --model mixtral_8x7b_instruct --file ./transcripts/awesome_nature_16.csv --batch_size 1 &> ./out/mixtral_16_1.out
python LLM.py --model mixtral_8x7b_instruct --file ./transcripts/awesome_nature_16.csv --batch_size 2 &> ./out/mixtral_16_2.out
python LLM.py --model mixtral_8x7b_instruct --file ./transcripts/awesome_nature_16.csv --batch_size 4 &> ./out/mixtral_16_4.out
python LLM.py --model mixtral_8x7b_instruct --file ./transcripts/awesome_nature_16.csv --batch_size 8 &> ./out/mixtral_16_8.out
python LLM.py --model mixtral_8x7b_instruct --file ./transcripts/awesome_nature_16.csv --batch_size 16 &> ./out/mixtral_16_16.out
python LLM.py --model mixtral_8x7b_instruct --file ./transcripts/awesome_nature_32.csv --batch_size 32 &> ./out/mixtral_32_32.out
python LLM.py --model llama2_70b_chat --file ./transcripts/awesome_nature_16.csv --batch_size 1 &> ./out/llama2_16_1.out
python LLM.py --model llama2_70b_chat --file ./transcripts/awesome_nature_16.csv --batch_size 2 &> ./out/llama2_16_2.out
python LLM.py --model llama2_70b_chat --file ./transcripts/awesome_nature_16.csv --batch_size 4 &> ./out/llama2_16_4.out
python LLM.py --model llama2_70b_chat --file ./transcripts/awesome_nature_16.csv --batch_size 8 &> ./out/llama2_16_8.out
python LLM.py --model llama2_70b_chat --file ./transcripts/awesome_nature_16.csv --batch_size 16 &> ./out/llama2_16_16.out
python LLM.py --model llama2_70b_chat --file ./transcripts/awesome_nature_32.csv --batch_size 32 &> ./out/llama2_32_32.out
