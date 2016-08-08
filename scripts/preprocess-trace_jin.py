"""
Preprocessing script for TRACE data.

"""

import os
import glob

def make_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

def tokenize(filepath, cp=''):
    dirpath = os.path.dirname(filepath)
    filepre = os.path.splitext(os.path.basename(filepath))[0]
    tokpath = os.path.join(dirpath, filepre + '.toks')
    cmd = ('java -cp %s edu.stanford.nlp.process.PTBTokenizer -preserveLines %s > %s'
        % (cp, filepath, tokpath))
    os.system(cmd)

def build_vocab(filepaths, dst_path, lowercase=True):
    vocab = set()
    for filepath in filepaths:
        with open(filepath) as f:
            for line in f:
                if lowercase:
                    line = line.lower()
                vocab |= set(line.split())
    with open(dst_path, 'w') as f:
        for w in sorted(vocab):
            f.write(w + '\n')

def split(filepath, dst_dir):
    with open(filepath) as datafile, \
         open(os.path.join(dst_dir, 'a.txt'), 'w') as afile, \
         open(os.path.join(dst_dir, 'b.txt'), 'w') as bfile,  \
         open(os.path.join(dst_dir, 'id.txt'), 'w') as idfile, \
         open(os.path.join(dst_dir, 'sim.txt'), 'w') as simfile:
            datafile.readline()
            for line in datafile:
                i, a, b, sim, ent = line.strip().split('\t')
                idfile.write(i + '\n')
                afile.write(a + '\n')
                bfile.write(b + '\n')
                simfile.write(sim + '\n')

def split_artifact(filepath, dst_dir):
    with open(filepath) as datafile, \
         open(os.path.join(dst_dir, 'id.txt'), 'w') as idfile, \
         open(os.path.join(dst_dir, 'sentence.txt'), 'w') as sentencefile:
        # data start from the first line
            for line in datafile:
                i, s = line.strip().split('\t')
                idfile.write(i + '\n')
                sentencefile.write(s + '\n')

def parse(dirpath, cp=''):
    tokenize(os.path.join(dirpath, 'a.txt'), cp=cp)
    tokenize(os.path.join(dirpath, 'b.txt'), cp=cp)

if __name__ == '__main__':
    print('=' * 80)
    print('Preprocessing TRACE dataset')
    print('=' * 80)


    trace_dir = os.path.join('/Users/Jinguo/Dropbox/Projects/2016_summer/tracenn/data/trace_new/')
    train_dir = os.path.join(trace_dir, 'train')
    dev_dir = os.path.join(trace_dir, 'dev')
    test_dir = os.path.join(trace_dir, 'test')
    source_dir = os.path.join(trace_dir, 'src_artf')
    target_dir = os.path.join(trace_dir, 'trg_artf')
    make_dirs([train_dir, dev_dir, test_dir, source_dir, target_dir])


    # split into separate files
    split(os.path.join(trace_dir, 'train.txt'), train_dir)
    split(os.path.join(trace_dir, 'validation.txt'), dev_dir)
    split(os.path.join(trace_dir, 'test.txt'), test_dir)
    split_artifact(os.path.join(trace_dir, 'SourceArtifact.txt'), source_dir)
    split_artifact(os.path.join(trace_dir, 'TargetArtifact.txt'), target_dir)
    # split(os.path.join(trace_dir, 'test_all_nosymbol.txt'), test_all_dir)


    # parse sentences
 #   parse(train_dir, cp=classpath)
 #   parse(dev_dir, cp=classpath)
 #   parse(test_dir, cp=classpath)

    # get vocabulary
 #   build_vocab(
 #       glob.glob(os.path.join(trace_dir, '*/*.toks')),
 #       os.path.join(trace_dir, 'vocab.txt'))
 #   build_vocab(
 #       glob.glob(os.path.join(trace_dir, '*/*.toks')),
 #       os.path.join(trace_dir, 'vocab-cased.txt'),
 #       lowercase=False)
