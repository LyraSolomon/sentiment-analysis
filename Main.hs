import Control.Arrow
import System.Directory
import Data.Char
import qualified Data.Text as T
import qualified Data.Text.IO as T
import Data.List
import qualified Data.Set as S
import qualified Data.Map.Strict as M
import qualified Data.IntMap.Strict as IM
import NLP.Stemmer
import Data.Ord
import Data.Function
import Data.SVM
import qualified Numeric.LinearAlgebra.Data as LA
import qualified Numeric.LinearAlgebra as LA

-- Ratings are embedded in the file names, e.g., 100_7.txt has a rating of 7/10. Extract it.
getRating :: (Fractional a, Read a) => String -> a
getRating = (\x -> (x-5.5)/4.5) . read . takeWhile (/= '.') . drop 1 . dropWhile (/= '_')

-- Get the rating and text from all files in a directory.
readSamples :: (Fractional a, Read a) => String -> IO [(a, T.Text)]
readSamples dir = mapM ((sequence .) $ getRating &&& T.readFile . (dir++)) . filter ((/= '.') . head) =<< getDirectoryContents dir

-- All of the training data, which is located across two directories.
trainingSamples :: (Fractional a, Read a) => IO [(a, T.Text)]
trainingSamples = do
  pos <- readSamples "aclImdb/train/pos/"
  neg <- readSamples "aclImdb/train/neg/"
  return $ takeEvery 5 (pos ++ neg)
validationSamples :: (Fractional a, Read a) => IO [(a, T.Text)]
validationSamples = do
  pos <- readSamples "aclImdb/train/pos/"
  neg <- readSamples "aclImdb/train/neg/"
  return $ takeEvery 10 (undefined:(pos ++ neg))

-- Words to ignore
stoplist :: S.Set T.Text
stoplist = S.fromList $ map T.pack ["i", "me", "my", "she", "her", "he", "him", "his", "they", "them", "their", "it", "who", "this", "that", "which",
  "", "br", "the", "a", "an", "is", "was", "be", "have", "are", "had", "were", "do", "did",
  "and", "or", "to", "in", "for", "with", "as", "one", "at", "by", "about", "from", "other", "into", "when", "would", "than",
  "also", "how", "becaus", "could", "after", "thing", "dont", "year", "two", "after", "peopl", "get", "where", "did", "off"]

minimalStoplist = S.fromList $ map T.pack ["the", "a", "an", "to", "is", "that", ""]
negateList = S.fromList $ map T.pack ["not", "isnt"]

-- Word splitter which ignores puctuation and doesn't care about capitalization or apostrophes.
words' :: T.Text -> [T.Text]
words' = map (stem' English) . filter (not . T.null) . T.split (not . isAlpha) . T.toLower . T.filter (/= '\'')

data PreFilterMode = Flip | Drop | Pass

-- Apply +/- modifier to word list
preFilter :: Num a => PreFilterMode -> [T.Text] -> [(a, T.Text)]
preFilter _ [] = []
preFilter Flip (w:ws)
  | w `S.member` negateList = alterHead (first negate) (preFilter Flip ws)
  | w `S.member` minimalStoplist = preFilter Flip ws
  | T.pack "un" `T.isPrefixOf` w = (-1, T.drop 2 w) : preFilter Flip ws
  | T.pack "nt" `T.isSuffixOf` w = (-1, T.dropEnd 2 w) : preFilter Flip ws
  | otherwise = (1, w) : preFilter Flip ws
  where alterHead f [] = []
        alterHead f (x:xs) = (f x) : xs
preFilter Drop (w:ws)
  | w `S.member` negateList = preFilter Drop ws
  | w `S.member` minimalStoplist = preFilter Drop ws
  | T.pack "un" `T.isPrefixOf` w = (1, T.drop 2 w) : preFilter Drop ws
  | T.pack "nt" `T.isSuffixOf` w = (1, T.dropEnd 2 w) : preFilter Drop ws
  | otherwise = (1, w) : preFilter Drop ws
preFilter Pass ws = zip (repeat 1) ws

-- Unify the number of occurences
bag :: Num a => [(a, T.Text)] -> [(a, T.Text)]
bag = map (sum . map fst &&& snd . head) . groupBy (curry $ uncurry (==) . both snd) . sortBy (comparing snd)

-- The stemming method from NLP.Stemmer, wrapped up to use Text insead of String
stem' :: Stemmer -> T.Text -> T.Text
stem' lang = T.pack . stem lang . T.unpack

-- The set of words in a dataset that meet the common threshold
common :: PreFilterMode -> Int -> [(a, T.Text)] -> S.Set T.Text
common mode thresh dataset = S.fromList . map snd . filter (\w -> fst w * thresh >= datalen) $ (df mode (`S.notMember` stoplist) dataset)
  where datalen = length dataset

-- Number of documents with each word not excluded by filter
df :: PreFilterMode -> (T.Text -> Bool) -> [(a, T.Text)] -> [(Int, T.Text)]
df mode wf = map (length &&& head) . group . sort . concatMap (map head . group . sort . filter wf . map snd . preFilter mode . words' . snd)

df' :: PreFilterMode -> (T.Text -> Bool) -> [(a, T.Text)] -> [(Float, T.Text)]
df' mode wf = map (log . balance . sum . map fst &&& snd . head) .
              groupBy ((==) `on` snd) . sortBy (comparing snd) .
              concatMap (bag . filter (wf . snd) . preFilter mode . words' . snd)

-- Calculates the Delta Inverse Document Frequency of the dataset
didf :: (Num a, Ord a) => PreFilterMode -> S.Set T.Text -> [(a, T.Text)] -> M.Map T.Text Float
didf mode legalWords dataset = uncurry (M.unionWith (+)) . both (M.fromList . map swap) .
    (                     df' mode (`S.member` legalWords) . filter ((>0) . fst) &&&
     map (first negate) . df' mode (`S.member` legalWords) . filter ((<0) . fst)) $ dataset

-- Converts a review into a vector for linear algebra
t2v :: PreFilterMode -> M.Map T.Text Int -> M.Map T.Text Double -> T.Text -> LA.Vector Double
t2v mode lut weights t = LA.fromList $ map (flip (IM.findWithDefault 0) (t2v' mode lut weights t)) [0 .. length lut - 1]

-- Converts a review into a vector for SVM, which is a different format
t2v' :: PreFilterMode -> M.Map T.Text Int -> M.Map T.Text Double -> T.Text -> IM.IntMap Double
t2v' mode lut weights t = IM.fromList . map (\(x, w) -> (lut M.! w, realToFrac $ weights M.! w * x)) .
                          filter ((`M.member` lut) . snd) . bag . preFilter mode . words' $ t

-- Converts a dataset into a term frequency matrix for linear algebra
data2m :: PreFilterMode -> M.Map T.Text Int -> M.Map T.Text Double -> [(a, T.Text)] -> LA.Matrix Double
data2m mode lut weights = LA.fromRows . map (t2v mode lut weights . snd)

-- Converts a matrix into a SVM problem
m2p :: [(Double, a)] -> LA.Matrix Double -> [(Double, IM.IntMap Double)]
m2p dataset = zip (map (clamp . fst) dataset) . map (\row -> IM.fromList $ map (id &&& (row LA.!)) [0 .. LA.size row - 1]) . LA.toRows

-- Applies a SVM model to a problem
checkProblem :: Model -> [(Double, IM.IntMap Double)] -> IO [(Double, Double)]
checkProblem model = sequence . map (sequence . second (predict model))

showReport :: (Ord a, Ord b, Num a, Num b) => [(a, b)] -> String
showReport results = show a ++ "/" ++ show n ++ " (" ++ show (100 * fromIntegral a / fromIntegral n) ++ "%)"
  where a = (length $ filter (\(a, b) -> (a>0) == (b>0)) results)
        n = (length results)

printReport :: (Ord a, Ord b, Num a, Num b) => (c, [(a, b)], [(a, b)]) -> IO ()
printReport (_, trainingResults, testingResults) = do
  putStrLn $ "\ttraining accuracy: " ++ showReport trainingResults ++ "\tvalidation accuracy: " ++ showReport testingResults

runExperiment :: (PreFilterMode, Int, Int, KernelType, Algorithm, [(Double, T.Text)], [(Double, T.Text)]) ->
                 IO (S.Set T.Text, [(Double, Double)], [(Double, Double)])
runExperiment (filterMode, commonThreshold, nSingularValues, svmKernel, svmAlgorithm, trainingDataset, testingDataset) = do
  let legalWords = common filterMode commonThreshold trainingDataset
  let weights = realToFrac <$> didf filterMode legalWords trainingDataset
  let wordIndexes = M.fromList $ zip (S.toList legalWords) [0..]
  let (u, s, v) = LA.thinSVD (data2m filterMode wordIndexes weights trainingDataset)
  let u' = u LA.?? (LA.All, LA.Take nSingularValues)
  let s' = LA.subVector 0 nSingularValues s
  let v' = v LA.?? (LA.All, LA.Take nSingularValues)
  let testingDataset' = m2p testingDataset $ data2m filterMode wordIndexes weights testingDataset LA.<> v'
  let trainingDataset' = m2p trainingDataset (u' LA.<> LA.diag s')
  svmModel <- withPrintFn (const $ return ()) $ train svmAlgorithm svmKernel trainingDataset'
  resultsTraining <- checkProblem svmModel trainingDataset'
  resultsValidation <- checkProblem svmModel testingDataset'
  return (legalWords, resultsTraining, resultsValidation)

main :: IO ()
main = do
  let thresh = 200
  let nsv = 300
  let kern = Linear --RBF 0.01
  let alg = CSvc 1

  dataset <- trainingSamples
  validationDataset <- validationSamples

  putStrLn "Our Method (Term Frequency Negation):"
  printReport =<< runExperiment (Flip, thresh, nsv, kern, alg, dataset, validationDataset)
  putStrLn "Stoplist & Stem Modifiers Without Negation:"
  printReport =<< runExperiment (Drop, thresh, nsv, kern, alg, dataset, validationDataset)
  putStrLn "Modifiers Treated Normally:"
  printReport =<< runExperiment (Pass, thresh, nsv, kern, alg, dataset, validationDataset)

-- Assorted utility functions

crop :: (Num a, Ord a) => a -> a
crop = max (-1) . min 1

clamp :: (Num a, Ord a) => a -> a
clamp x = if x > 0 then 1 else -1

balance :: (Ord a, Fractional a) => a -> a
balance x = if x > 0 then 1 + x else 1 / (1 - x)

takeEvery :: Int -> [a] -> [a]
takeEvery n xs = case drop (n-1) xs of (y:ys) -> y : takeEvery n ys; [] -> []

dropEvery :: Int -> [a] -> [a]
dropEvery n xs = case splitAt (n-1) xs of (x, (y:ys)) -> x ++ dropEvery n ys; (x, []) -> x

r2 :: Floating a => [(a, a)] -> (a, (a, a))
r2 points = let n = fromIntegral (length points)
                ex = sum (map fst points) / n
                ey = sum (map snd points) / n
                sx = sqrt $ sum (map ((^2) . ((-)ex) . fst) points) / n
                sy = sqrt $ sum (map ((^2) . ((-)ey) . snd) points) / n
                cov = sum (map (\(x, y) -> (x - ex) * (y - ey)) points) / n
            in (cov / (sx * sy), (sx / sy, ex * sy / sx - ey))

both :: (a -> b) -> (a, a) -> (b, b)
both f (x, y) = (f x, f y)

swap :: (a, b) -> (b, a)
swap (x, y) = (y, x)
