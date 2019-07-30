from tqdm import tqdm
input_path = "./test_word_per_line.txt"
output_path = "./test_cleaned.txt"

curr_id = ""
curr_sent = []
with open(output_path, mode="wt", encoding="utf-8") as f:
	lines = open(input_path).readlines()
	for idx, line in tqdm(enumerate(lines)):
		if idx == 0:
			continue
		line = line.strip()
		line_id = line.split(",")[0][:3]
		if line_id != curr_id:
			if len(curr_sent) > 0:
				f.write("{},{}\n".format(curr_id, ' '.join(curr_sent)))
			curr_id = line_id
			curr_sent = []
		curr_sent.append(line.split(",")[-1])
		if idx == len(lines) - 1:
			f.write("{},{}\n".format(curr_id, ' '.join(curr_sent)))
