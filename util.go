package fasttext

import (
	"bytes"
	"encoding/binary"
)

func vecToBytes(vec []float32, order binary.ByteOrder) []byte {
	buf := new(bytes.Buffer)
	for _, v := range vec {
		binary.Write(buf, order, v)
	}
	return buf.Bytes()
}

func bytesToVec(data []byte, order binary.ByteOrder) ([]float32, error) {
	size := len(data) / 4
	vec := make([]float32, size)
	buf := bytes.NewReader(data)
	var v float32
	for i := range vec {
		if err := binary.Read(buf, order, &v); err != nil {
			return nil, err
		}
		vec[i] = v
	}
	return vec, nil
}
