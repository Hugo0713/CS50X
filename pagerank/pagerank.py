import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    prob = {p: 0 for p in corpus}
    if page not in corpus:
        for p in corpus:
            prob[p] = 1 / len(corpus)
        return prob
    for p in corpus[page]:
        prob[p] += damping_factor / len(corpus[page])
    for p in corpus:
        prob[p] += (1 - damping_factor) / len(corpus)
    return prob
    # raise NotImplementedError


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    values = {p: 0 for p in corpus}
    page = random.choice(list(corpus.keys()))
    for i in range(n):
        prob = transition_model(corpus, page, damping_factor)
        page = random.choices(list(prob.keys()), weights = list(prob.values()), k=1)[0]
        values[page] += 1
    for page in values:
        values[page] /= n
    return values


    # raise NotImplementedError


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    N = len(corpus)
    values = {p: 1 / N for p in corpus}
    values_new = {p: 0 for p in corpus}
    while True:
        dangling_sum = 0
        for dangling_page in values:
            if len(corpus[dangling_page]) == 0:
                dangling_sum += values[dangling_page]
        for page in list(values.keys()):
            link_sum = 0
            links = []
            for page_link in list(values.keys()):
                if page in corpus[page_link]:
                    links.append(page_link)
            for p in links:
                numlinks = len(corpus[p]) 
                link_sum += values[p] / numlinks
            
            values_new[page] = (1 - damping_factor) / N + damping_factor * (link_sum + dangling_sum / N)
        if all(abs(values[page] - values_new[page]) < 0.001 for page in values):
            break
        values = values_new.copy()

    return values
    


    # raise NotImplementedError


if __name__ == "__main__":
    main()
