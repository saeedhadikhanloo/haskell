{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE ScopedTypeVariables #-}

import Control.Monad (replicateM_)
import qualified Data.Vector as Vector
import Data.Vector (Vector)
import TensorFlow.Core (Scalar (..), Tensor, Value, encodeTensorData, feed)
import TensorFlow.GenOps.Core (square)
import TensorFlow.Gradient (gradients)
import TensorFlow.Minimize (gradientDescent, minimizeWith)
import TensorFlow.Ops (add, constant, mul, placeholder, reduceSum, sub)
import TensorFlow.Session (run, runSession, runWithFeeds)
import TensorFlow.Variable (Variable, initializedVariable, readValue)

import TensorFlow.Tensor

import TensorFlow.Types (fromTensorTypeList)

-- import Graphics.Rendering.Chart.Backend.Cairo
-- import Graphics.Rendering.Chart.Easy hiding (makeAxis)

makeAxis :: [Float] -> [Float] -> [(Float, Float)]
makeAxis axis1 axis2 = [(t, t') | t <- axis1, t' <- axis2]

computeGd :: (Float, Float) -> IO (Tensor Value Float)
computeGd (x, y) = do
    gdio <- runSession $ do
        (xv :: Variable Float) <- initializedVariable 0
        (yv :: Variable Float) <- initializedVariable 1
        let xvr = readValue xv
            yvr = readValue yv
            linear_model = mul xvr yvr
        gd <- gradients linear_model [xv]
        let list = gd
        return $ list !! 0
    print (gdio)
    return gdio
  -- xv <- initializedVariable 1
  -- yv <- initializedVariable 2
  -- (Scalar xv_learned, Scalar yv_learned) <- run (readValue xv, readValue yv)
-- let additionNode = node1 `add` node2
-- vec <- run additionNode
--return (list !! 0 , list !! 0)

main :: IO ()
main = do
  let n = 1
      xs = (\x -> x / n) <$> [(- 5 * n :: Float) .. 5 * n]
      ys = (\x -> x / n) <$> [(- 5 * n :: Float) .. 5 * n]
      grid = makeAxis xs ys
      fileName = "gd-field.png"
  vectors <- mapM computeGd grid
  putStrLn "\nDone\n"
-- toFile (FileOptions (1500,1500) PNG) fileName $ do
--  setColors [opaque black]
--  plot $ vectorField grid vectors
-- putStrLn $ "\nCheck out " ++ fileName ++ "\n"
-- where
--  vectorField grid vectors = fmap plotVectorField $ liftEC $ do
--    c <- takeColor
--    plot_vectors_values .= zip grid vectors
--    plot_vectors_style . vector_line_style . line_width .= 1
--    plot_vectors_style . vector_line_style . line_color .= c
--    plot_vectors_style . vector_head_style . point_radius .= 0.0
