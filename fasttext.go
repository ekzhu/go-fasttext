/*
Package fasttext provides a simple wrapper for Facebook
fastText dataset (https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md).
It allows fast look-up of word embeddings from persistent data store (Sqlite3).

Installation

	go get -u github.com/ekzhu/go-fasttext

After downloading a .vec data file from the fastText project,
you can initialize the Sqlite3 database (in your code):

	ft := NewFastText("/path/to/sqlite3/file")
	err := ft.BuilDB("/path/to/word/embedding/.vec/file")

This will create a new file on your disk for the Sqlite3 database.
Once the above step is finished, you can start looking up word embeddings
(in your code):

	emb, err := ft.GetEmb("king")
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println(emb)

Each word embedding vector is a slice of float64.

Note that you only need to initialize the Sqlite3 database once.
The next time you use it you can skip the call to BuildDB.
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

	_ "github.com/mattn/go-sqlite3"
)

const (
	// Table name used in Sqlite
	TableName = "fasttext"
	Dim       = 300
)

var (
	ErrNoEmbFound = errors.New("No embedding found for the given word")
	// TODO: parametrize byte order
	ByteOrder = binary.BigEndian
)

type FastText struct {
	db        *sql.DB
	tablename string
	byteOrder binary.ByteOrder
}

// wordEmb is a pair of word and its embedding vector.
type wordEmb struct {
	Word string
	Vec  []float64
}

// Start a new FastText session given the location
// of the Sqlite3 database file.
func NewFastText(dbFilename string) *FastText {
	db, err := sql.Open("sqlite3", dbFilename)
	if err != nil {
		panic(err)
	}
	return &FastText{
		db:        db,
		tablename: TableName,
		byteOrder: ByteOrder,
	}
}

// Close must be called before finishing using FastText
func (ft *FastText) Close() error {
	return ft.db.Close()
}

// GetEmb returns the word embedding of the given word.
func (ft *FastText) GetEmb(word string) ([]float64, error) {
	var binVec []byte
	err := ft.db.QueryRow(fmt.Sprintf(`
	SELECT emb FROM %s WHERE word=?;
	`, ft.tablename), word).Scan(&binVec)
	if err == sql.ErrNoRows {
		return nil, ErrNoEmbFound
	}
	if err != nil {
		panic(err)
	}
	return bytesToVec(binVec, ft.byteOrder)
}

// BuilDB initialize the Sqlite database by importing the word embeddings
// from the .vec file downloaded from
// https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md
func (ft *FastText) BuildDB(wordEmbFile io.Reader) error {
	_, err := ft.db.Exec(fmt.Sprintf(`
	CREATE TABLE %s (
		word TEXT UNIQUE,
		emb BLOB
	);
	`, ft.tablename))
	if err != nil {
		return err
	}
	stmt, err := ft.db.Prepare(fmt.Sprintf(`
	INSERT INTO %s(word, emb) VALUES(?, ?);
	`, ft.tablename))
	if err != nil {
		return err
	}
	defer stmt.Close()
	for emb := range readwordEmbdFile(wordEmbFile) {
		binVec := vecToBytes(emb.Vec, ft.byteOrder)
		if _, err := stmt.Exec(emb.Word, binVec); err != nil {
			return err
		}
	}
	// Indexing on words
	_, err = ft.db.Exec(fmt.Sprintf(`
	CREATE INDEX ind_word ON %s(word);
	`, ft.tablename))
	if err != nil {
		return err
	}
	return nil
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
