import glob

def main():

    filenames = glob.glob('1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/*')

    number = 0
    for filename in filenames:
        number += 1
        print("Processing file number %d." % number, end='\r', flush=True)
        with open(filename, 'r') as infile:
            for line in infile:
                sentence = line.split()
                length = len(sentence)

                if length < 4 or length > 10:
                    continue

                with open('data/bwb%02d' % length, 'a') as outfile:
                    outfile.write(' '.join(sentence).lower())
                    outfile.write('\n')

    print()

if __name__ == "__main__":

    main()
