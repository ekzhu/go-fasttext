package fasttext

import (
	"os"
	"testing"
)

func Test_BuildDB_and_GetEmb(t *testing.T) {
	ft := NewFastText(":memory:")
	defer ft.Close()

	file, err := os.Open("./testdata/wiki.en.vec")
	if err != nil {
		t.Error(err)
	}
	defer file.Close()
	err = ft.BuildDB(file)
	if err != nil {
		t.Error(err)
	}

	words := []string{"has", "but", "page", "#"}
	for _, word := range words {
		emb, err := ft.GetEmb(word)
		if err != nil {
			t.Error(err)
		}
		t.Log(emb)
	}

	notExist := []string{"NotExist1", "Happiness"}
	for _, word := range notExist {
		_, err := ft.GetEmb(word)
		if err != ErrNoEmbFound {
			t.Error("Should return not found")
		}
	}
}
