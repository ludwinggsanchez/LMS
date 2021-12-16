package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"math/rand"
	"os"
	"strconv"
	"time"

	"gonum.org/v1/gonum/mat"
)

type LMS struct {
	mu      float64
	weights []float64
	iterNum int
}

func main() {
	//read data
	irisMatrix := [][]string{}
	iris, err := os.Open("data0.csv")
	if err != nil {
		panic(err)
	}
	defer iris.Close()

	reader := csv.NewReader(iris)
	reader.Comma = ','
	reader.LazyQuotes = true
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			panic(err)
		}
		irisMatrix = append(irisMatrix, record)
	}

	//separate data into explaining and explained variables
	X := [][]float64{}
	Y := []float64{}
	for _, data := range irisMatrix {

		//convert str slice to float slice
		temp := []float64{}
		for _, i := range data[:2] {
			parsedValue, err := strconv.ParseFloat(i, 64)
			if err != nil {
				panic(err)
			}
			temp = append(temp, parsedValue)
		}

		temp = append([]float64{1}, temp...)
		// X = [x0, x1, x2]
		//explaining
		X = append(X, temp)
		//explained
		if data[2] == "-1" {
			Y = append(Y, -1.0)
		} else {
			Y = append(Y, 1.0)
		}
	}
	rand.Seed(time.Now().UnixNano())
	//training
	lms := LMS{0.01, []float64{}, 100}
	lms.fit(X, Y)

}

func activate(linearCombination float64) float64 {
	if linearCombination > 0 {
		return 1.0
	} else {
		return -1.0
	}
}

// predict returns the models internal value for a given input vector and current weights vector
func (p *LMS) predict(x []float64) float64 {
	w := mat.NewDense(1, len(p.weights), p.weights)
	xx := mat.NewDense(1, len(x), x)

	var c mat.Dense
	c.Mul(w, xx.T())
	return c.RawMatrix().Data[0]
}

// update updated the weights according to the LMS algorithm
func (p *LMS) update(x []float64, y float64) (errr float64) {
	w := mat.NewDense(1, len(p.weights), p.weights)
	xx := mat.NewDense(1, len(x), x)
	// g = w·x
	var g mat.Dense
	g.Mul(w, xx.T())

	// t
	yExpanded := mat.NewDense(1, 1, []float64{y})
	yExpanded.Scale(-1, yExpanded)

	// e = g - t
	var e mat.Dense
	e.Add(&g, yExpanded)

	// mu·e·x
	var muex mat.Dense
	e.Scale(p.mu, &e)
	muex.Mul(&e, xx)
	muex.Scale(-1, &muex)

	// w = w - mu·e·x
	w.Add(w, &muex)
	p.weights = w.RawMatrix().Data

	return e.At(0, 0)
}

func (p *LMS) fit(X [][]float64, Y []float64) {
	// initialize the weights

	data := [][]string{}
	data = append(data, []string{
		"iteración", "x0", "x1", "x2", "y",
		"w0", "w1", "w2", "y'", "update?", "err",
		"w0'", "w1'", "w2'", "m", "b",
	})

	//update weights by data
	p.weights = []float64{}

	// Init weights
	for range X[0] {
		p.weights = append(p.weights, rand.NormFloat64())
	}

	errr := []string{}
	var stop bool
	for iter := 0; iter < p.iterNum; iter++ {
		if stop {
			break
		}
		for i := 0; i < len(X); i++ {
			fmt.Println("iter", iter, i)
			y_pred := p.predict(X[i])
			if i < int(float64(len(X))*float64(0.7)) {
				row := []string{
					fmt.Sprintf("%.0f", float64(i)),
					fmt.Sprintf("%.4f", X[i][0]),
					fmt.Sprintf("%.4f", X[i][1]),
					fmt.Sprintf("%.4f", X[i][2]),
					fmt.Sprintf("%.4f", Y[i]),
					fmt.Sprintf("%.4f", p.weights[0]),
					fmt.Sprintf("%.4f", p.weights[1]),
					fmt.Sprintf("%.4f", p.weights[2]),
					fmt.Sprintf("%.4f", activate(y_pred)),
					strconv.FormatBool(y_pred*Y[i] < 0),
				}

				if y_pred*Y[i] < 0 {
					err := p.update(X[i], Y[i])

					errr = append(errr, fmt.Sprintf("%.4f", err))
					row = append(row, fmt.Sprintf("%.4f", err))
					row = append(row, fmt.Sprintf("%.4f", p.weights[0]))
					row = append(row, fmt.Sprintf("%.4f", p.weights[1]))
					row = append(row, fmt.Sprintf("%.4f", p.weights[2]))
					if c1, c2 := p.validate(X, Y, false); c1 > 0.5 && c2 > 0.5 {
						stop = true
						break
					}
				} else {
					row = append(row, "")
					row = append(row, "")
					row = append(row, "")
					row = append(row, "")
				}
				m := -(p.weights[0] / p.weights[2]) / (p.weights[0] / p.weights[1])
				b := -p.weights[0] / p.weights[2]
				row = append(row, fmt.Sprintf("%.4f", m))
				row = append(row, fmt.Sprintf("%.4f", b))
				data = append(data, row)
			}
		}
	}
	m := -p.weights[1] / p.weights[2]
	b := -p.weights[0] / p.weights[2]
	fmt.Println("y = x * ", m, " + ", b, ";")
	p.validate(X, Y, true)
	data = append(data, errr)

	f, err := os.Create("results.csv")
	defer f.Close()

	if err != nil {
		log.Fatalln("failed to open file", err)
	}

	w := csv.NewWriter(f)
	err = w.WriteAll(data)

	if err != nil {
		log.Fatal(err)
	}
}

// validate tests the model given a weights vector
func (p *LMS) validate(X [][]float64, Y []float64, validacion bool) (C1out float64, C2out float64) {

	C1Count := 0
	C2Count := 0
	vP := 0
	vN := 0
	fP := 0
	fN := 0
	C1Out := 0
	C2Out := 0
	init := 0
	end := 0

	if validacion {
		init = int(float64(len(X)) * 0.7)
		end = len(X)
	} else {
		init = 0
		end = int(float64(len(X)) * 0.7)
	}

	for i := init; i < end; i++ {
		y_pred := p.predict(X[i])
		if Y[i] > 0 {
			C1Count++
		} else {
			C2Count++
		}

		if y_pred > 0 {
			C1Out++
		} else {
			C2Out++
		}

		if y_pred*Y[i] < 0 {
			if y_pred > 0 {
				fP++
			} else {
				fN++
			}
		} else {
			if Y[i] > 0 {
				vP++
			} else {
				vN++
			}
		}
	}
	fmt.Println(C1Count, C2Count, "C1: ", C1Out, "C2: ", C2Out, "vP:", vP, "fP:", fP, "vN:", vN, "fN:", fN)

	return float64(vP) / float64(C1Count), float64(vN) / float64(C2Count)
}
