package fasttext

import (
	"bytes"
	"encoding/binary"
)

func vecToBytes(vec []float64, order binary.ByteOrder) []byte {
	buf := new(bytes.Buffer)
	for _, v := range vec {
		binary.Write(buf, order, float32(v))
	}
	return buf.Bytes()
}

func bytesToVec(data []byte, order binary.ByteOrder) ([]float64, error) {
	size := len(data) / 4
	vec := make([]float64, size)
	buf := bytes.NewReader(data)
	var v float32
	for i := range vec {
		if err := binary.Read(buf, order, &v); err != nil {
			return nil, err
		}
		vec[i] = float64(v)
	}
	return vec, nil
}
