import System.Random
import Data.Matrix hiding (trace)

-- Nonlin function
nonlin:: Double -> Double
nonlin x = x * (1.0 - x)

-- Sigmoid function
sigmoid:: Double -> Double
sigmoid x = 1.0 / (1.0 + (exp (negate x)))

--Make a list of random numbers
randomList :: (Random a) => Int -> [a]
randomList seed = randoms (mkStdGen seed)

-- Orginal example data
-------------------------------------------------------------------
-- Input dataset
input_dataset:: Matrix Double
input_dataset = fromLists [[0,0,1], [0,1,1], [1,0,1], [1,1,1]]

-- Labels
answers:: Matrix Double
answers = transpose $  fromLists [[0,0,1,1]]
-------------------------------------------------------------------

--Make starting weights
syn0:: Matrix Double
syn0 = fromLists [[x] | x<- take (ncols input_dataset) (randomList 42 :: [Double])]
--Function to update the weights
nextGeneration:: Matrix Double -> Matrix Double -> (Matrix Double, Matrix Double) -> (Matrix Double, Matrix Double)
nextGeneration l0 labels (syn0, _) = (new_syn0, l1)
    where
        -- Forward propagation
        l1 = fmap sigmoid $ multStd l0 syn0 
        -- How much did we miss?
        l1_error =  elementwise (-) labels l1
        -- Multiply how much we missed by the slope of the sigmoid at the values in l1
        l1_delta = elementwise (*) l1_error $ fmap nonlin l1 
        -- Update weights
        new_syn0 = elementwise (+) syn0 $ multStd (transpose l0) l1_delta 

main = putStrLn message
   where 
       result = iterate (nextGeneration input_dataset answers ) (syn0, fromLists [] )!!10000
       message = "Output after traning:\n" ++ show (snd result)




