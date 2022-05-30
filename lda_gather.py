import lda_prepare
import pathlib, random



if __name__ == '__main__':
    pth = pathlib.Path("data.txt")
    while True:
        with open(str(pth), mode="w", encoding="utf-8") as f:
            cycle = 0
            systopics = random.randrange(20, 100)
            #chunk_size = random.randrange(20, 100)
            #passes = random.randrange(20, 100)
            #iterations = random.randrange(20, 100)
            
            perplexity, coherence = lda_prepare.prepare(topics=systopics)
            data = f"Cycle {cycle}: Perplexity: {perplexity}, Coherence: {coherence}"

            print(data)
            f.write(data)
            cycle += 1