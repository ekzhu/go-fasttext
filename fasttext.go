/*
Package fasttext provides a simple wrapper for Facebook
fastText dataset (https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md).
It allows fast look-up of word embeddings from persistent data store (SQLite3).

Installation

	go get -u github.com/ekzhu/go-fasttext

After downloading a .vec data file from the fastText project,
you can initialize the SQLite3 database (in your code):

	import (
		_ "github.com/mattn/go-sqlite3"
		"github.com/ekzhu/go-fasttext"
	)
	...
	ft := fasttext.NewFastText("/path/to/sqlite3/file")
	err := ft.BuilDB("/path/to/word/embedding/.vec/file")

This will create a new file on your disk for the SQLite3 database.
Once the above step is finished, you can start looking up word embeddings
(in your code):

	emb, err := ft.GetEmb("king")
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println(emb)

Each word embedding vector is a slice of float64 with length of 300.

Note that you only need to initialize the SQLite3 database once.
The next time you use it you can skip the call to BuildDB.

For faster querying during runtime, you can use an in-memory database.

	ft := NewFastTextInMem("/path/to/sqlite3/file")

This creates an in-memory SQLite3 database which is a copy of the
on-disk one. Using the in-memory version makes query time much faster,
but takes a few minutes to load the database.
*/
package fasttext

import (
	"bufio"
	"database/sql"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"strconv"
	"strings"
)

const (
	// TableName used in SQLite3
	TableName = "fasttext"
	// Dim is the number of dimensions in FastText word embedding vectors
	Dim = 300
)

var (
	// ErrNoEmbFound ...
	ErrNoEmbFound = errors.New("No embedding found for the given word")
	// ByteOrder is for the serialization of the embedding vector in
	// SQLite3 database.
	ByteOrder = binary.BigEndian
)

// The FastText session.
// In multi-thread setting, each thread must have its own copy of
// FastText session. A single FastText session cannot be shared
// among multiple threads.
type FastText struct {
	db *sql.DB
}

// NewFastText starts a new FastText session given the location
// of the SQLite3 database file.
func NewFastText(dbFilename string) *FastText {
	db, err := sql.Open("sqlite3", dbFilename)
	if err != nil {
		panic(err)
	}
	return &FastText{
		db: db,
	}
}

// NewFastTextInMem creates a new FastText session that uses
// an in-memory database for faster query time.
// The on-disk SQLite3 database (given by dbFilename) will be loaded into
// an in-memory SQLite3 database in this function, which
// will take a few miniutes to finish.
func NewFastTextInMem(dbFilename string) *FastText {
	db, err := sql.Open("sqlite3", "file::memory:?cache=shared")
	_, err = db.Exec(fmt.Sprintf(`ATTACH DATABASE '%s' AS disk;`, dbFilename))
	if err != nil {
		panic(err)
	}
	_, err = db.Exec(`CREATE TABLE fasttext AS SELECT * FROM disk.fasttext;`)
	if err != nil {
		panic(err)
	}
	_, err = db.Exec(`CREATE INDEX inx_ft ON fasttext(word);`)
	if err != nil {
		panic(err)
	}
	return &FastText{
		db: db,
	}
}

// Close must be called before finishing using this FastText
// session.
func (ft *FastText) Close() error {
	return ft.db.Close()
}

// GetEmb returns the word embedding of the given word.
func (ft *FastText) GetEmb(word string) ([]float64, error) {
	var binVec []byte
	err := ft.db.QueryRow(`SELECT emb FROM fasttext WHERE word=?;`, word).Scan(&binVec)
	if err == sql.ErrNoRows {
		return nil, ErrNoEmbFound
	}
	if err != nil {
		panic(err)
	}
	return bytesToVec(binVec, ByteOrder)
}

// BuildDB initialize the SQLite3 database by importing the word embeddings
// from the .vec file downloaded from
// https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md
func (ft *FastText) BuildDB(wordEmbFile io.Reader) error {
	_, err := ft.db.Exec(`
	CREATE TABLE fasttext(
		word TEXT UNIQUE,
		emb BLOB
	);`)
	if err != nil {
		return err
	}
	stmt, err := ft.db.Prepare(`INSERT INTO fasttext(word, emb) VALUES(?, ?);`)
	if err != nil {
		return err
	}
	defer stmt.Close()
	for emb := range readwordEmbdFile(wordEmbFile) {
		binVec := vecToBytes(emb.Vec, ByteOrder)
		if _, err := stmt.Exec(emb.Word, binVec); err != nil {
			return err
		}
	}
	// Indexing on words
	_, err = ft.db.Exec(`CREATE INDEX ind_word ON fasttext(word);`)
	if err != nil {
		return err
	}
	return nil
}

type wordEmb struct {
	Word string
	Vec  []float64
}

func readwordEmbdFile(wordEmbFile io.Reader) chan *wordEmb {
	out := make(chan *wordEmb)
	go func() {
		defer close(out)
		scanner := bufio.NewScanner(wordEmbFile)
		var embSize int
		var line int
		for scanner.Scan() {
			line++
			data := scanner.Text()
			if embSize == 0 {
				var err error
				embSize, err = strconv.Atoi(strings.Split(data, " ")[1])
				if err != nil {
					panic(err)
				}
				continue
			}
			// Get the word
			items := strings.SplitN(data, " ", 2)
			word := items[0]
			if word == "" {
				word = " "
			}
			// Get the vec
			vecStrs := strings.Split(strings.TrimSpace(items[1]), " ")
			if len(vecStrs) != embSize {
				msg := fmt.Sprintf("Embedding vec size not same: expected %d, got %d. Loc: line %d, word %s",
					embSize, len(vecStrs), line, word)
				panic(msg)
			}
			vec := make([]float64, embSize)
			for i := 0; i < embSize; i++ {
				sf, err := strconv.ParseFloat(vecStrs[i], 64)
				if err != nil {
					panic(err)
				}
				vec[i] = sf
			}
			out <- &wordEmb{
				Word: word,
				Vec:  vec,
			}
		}
		if err := scanner.Err(); err != nil {
			panic(err)
		}
	}()
	return out
}
