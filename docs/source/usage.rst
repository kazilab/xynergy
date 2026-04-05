###############
 Using Xynergy
###############

*******************
 What Xynergy does
*******************

Xynergy allows you to calculate drug synergy from dose-response matrices
with minimal data. It does this by imputing the missing data, then
calculating synergy as normal.

.. list-table::

   -  -  .. figure:: ./_static/minimal_matrix.png
            :width: 200px

            The data you have

      -  .. figure:: ./_static/full_matrix.png
            :width: 200px

            The data you want

      -  .. figure:: ./_static/imputed_matrix.png
            :width: 200px

            The data Xynergy imputes

***********************
 Example data overview
***********************

Xynergy bundles a small real workbook example at
`xynergy/example_data/data.xlsx`. The helper below reads that workbook,
renames the columns to Xynergy's canonical names, and converts the
worksheet's viability-style percentages into inhibition-style response
percentages.

.. code:: python

   from xynergy.example import load_example_data

   example_data = load_example_data()

`example_data` contains a single drug pair measured at six dose levels.
Only the off-axis single-agent doses plus the dose-matched diagonal are
observed, so the data still have the sparse "minimum combination data"
shape shown in the left figure above.

The bundled workbook has:

- One experiment
- One drug pair: Venetoclax + Volasertib
- Six dose levels on each axis
- Sixteen observed combinations, which expand to a 6 x 6 matrix after tidying

The code snippets below now use this bundled workbook example. Some
printed output tables later in the page are illustrative summaries from
older runs and may not match the workbook values exactly.

If you want to work with the raw workbook columns instead, use
`load_example_data(raw=True)`.

**************
 Tidying data
**************

The workbook example already has one response per row, so `tidy` mainly
does three things for us here:

- Checks the input columns
- Creates `experiment_id`
- Expands the sparse axis-plus-diagonal observations into the full matrix,
  filling missing combinations with `null`

Because `load_example_data()` already returns the preferred column names
`dose_a`, `dose_b`, and `response`, we only need to specify the columns
that identify an experiment:

.. code:: python

   from xynergy.tidy import tidy

   clean_data = tidy(
       example_data,
       experiment_cols=[
           "experiment_source_id",
           "line",
           "drug_a",
           "drug_b",
           "pair_index",
       ],
   )

For the bundled workbook, `tidy` expands 16 observed rows to 36 matrix
rows, adding 20 `null` placeholders for the unmeasured interior
combinations.

If you do not care about preserving the metadata columns, the bundled
example also works as a single-experiment dataset:

.. code:: python

   tidy(example_data)

Tidying is just the first step, though. Our next step is to pre-impute
the missing values.

****************
 Pre-imputation
****************

Let's start simple. We'll use the iterative imputer from `scikit-learn
<https://scikit-learn.org/stable/index.html>`_ to start. Don't worry
about the `use_single_drug_response_data` argument yet.

.. code:: python

   from xynergy.impute import pre_impute, post_impute

   imputed = pre_impute(
       clean_data,
       method="IterativeImputer",
       use_single_drug_response_data=False,
       experiment_cols="experiment_id",
       dose_cols=["dose_a", "dose_b"],
       response_col="response",
   )

Actually, since we ran our data through `tidy`, our column names have
been set to be the defaults of downstream functions - so as a bonus we
can omit many of these arguments:

.. code:: python

   pre_impute(
       clean_data,
       method="IterativeImputer",
       use_single_drug_response_data=False,
   )

.. code::

   ┌───────────────┬────────┬────────┬──────────┬──────────────┐
   │ experiment_id ┆ dose_a ┆ dose_b ┆ response ┆ resp_imputed │
   │ ---           ┆ ---    ┆ ---    ┆ ---      ┆ ---          │
   │ u32           ┆ f64    ┆ f64    ┆ f64      ┆ f64          │
   ╞═══════════════╪════════╪════════╪══════════╪══════════════╡
   │ 1             ┆ 0.0    ┆ 0.0    ┆ 0.0      ┆ 0.0          │
   │ 1             ┆ 0.0    ┆ 0.0    ┆ 0.0      ┆ 0.0          │
   │ 1             ┆ 0.0    ┆ 0.0    ┆ 0.0      ┆ 0.0          │
   │ 1             ┆ 0.0    ┆ 0.0    ┆ 0.0      ┆ 0.0          │
   │ 1             ┆ 0.0    ┆ 0.0    ┆ 0.0      ┆ 0.0          │
   │ …             ┆ …      ┆ …      ┆ …        ┆ …            │
   │ 2             ┆ 1000.0 ┆ 3.16   ┆ null     ┆ 75.86        │
   │ 2             ┆ 1000.0 ┆ 56.23  ┆ null     ┆ 78.35        │
   │ 2             ┆ 1000.0 ┆ 1000.0 ┆ 99.99    ┆ 99.99        │
   │ 2             ┆ 1000.0 ┆ 1000.0 ┆ 99.99    ┆ 99.99        │
   │ 2             ┆ 1000.0 ┆ 1000.0 ┆ 99.99    ┆ 99.99        │
   └───────────────┴────────┴────────┴──────────┴──────────────┘

.. NOTE::

   I've removed the experimental columns - `line`, `name_a`, `name_b`,
   and `replicate` - to ensure all the data fit easily on the screen.
   However, they would normally be returned.

You'll notice a new column, `resp_imputed`, with values that are `null`
in the same row of the `response` column. These are our (pre-)imputed
values!

For some algorithms, we can improve imputation accuracy by including
columns that contain the response we would get if we JUST added `drug_a`
or `drug_b`, for instance (refer to the figure below).

   .. figure:: ./_static/uncombined_drugs.png

      Intuition of `use_single_drug_response_data`

We can do that automatically by setting
`use_single_drug_response_data=True` (the default, so we'll just forgo
setting it):

.. code:: python

   pre_impute(
       clean_data,
       method="IterativeImputer",
   )

.. code::

   ┌───────────────┬────────┬────────┬──────────┬─────────────┬─────────────┬──────────────┐
   │ experiment_id ┆ dose_a ┆ dose_b ┆ response ┆ dose_a_resp ┆ dose_b_resp ┆ resp_imputed │
   │ ---           ┆ ---    ┆ ---    ┆ ---      ┆ ---         ┆ ---         ┆ ---          │
   │ u32           ┆ f64    ┆ f64    ┆ f64      ┆ f64         ┆ f64         ┆ f64          │
   ╞═══════════════╪════════╪════════╪══════════╪═════════════╪═════════════╪══════════════╡
   │ 1             ┆ 0.0    ┆ 0.0    ┆ 0.0      ┆ 0.04        ┆ 0.0         ┆ 0.0          │
   │ 1             ┆ 0.0    ┆ 0.0    ┆ 0.0      ┆ 0.04        ┆ 0.0         ┆ 0.0          │
   │ 1             ┆ 0.0    ┆ 0.0    ┆ 0.0      ┆ 0.04        ┆ 0.0         ┆ 0.0          │
   │ 1             ┆ 0.0    ┆ 0.0    ┆ 0.0      ┆ 0.04        ┆ 0.0         ┆ 0.0          │
   │ 1             ┆ 0.0    ┆ 0.0    ┆ 0.0      ┆ 0.04        ┆ 0.0         ┆ 0.0          │
   │ …             ┆ …      ┆ …      ┆ …        ┆ …           ┆ …           ┆ …            │
   │ 2             ┆ 1000.0 ┆ 3.16   ┆ null     ┆ 98.92       ┆ 23.23       ┆ 83.92        │
   │ 2             ┆ 1000.0 ┆ 56.23  ┆ null     ┆ 98.92       ┆ 85.1        ┆ 123.07       │
   │ 2             ┆ 1000.0 ┆ 1000.0 ┆ 99.99    ┆ 98.92       ┆ 99.08       ┆ 99.99        │
   │ 2             ┆ 1000.0 ┆ 1000.0 ┆ 99.99    ┆ 98.92       ┆ 99.08       ┆ 99.99        │
   │ 2             ┆ 1000.0 ┆ 1000.0 ┆ 99.99    ┆ 98.92       ┆ 99.08       ┆ 99.99        │
   └───────────────┴────────┴────────┴──────────┴─────────────┴─────────────┴──────────────┘

These results are *ok*, and they certainly were quick to get. However,
if we have the time we can significantly improve our accuracy of
imputation using `XGBoost
<https://xgboost.readthedocs.io/en/release_3.0.0/#>`_ regression:

.. code:: python

   imputed = pre_impute(clean_data, method="XGBR")
   imputed

.. code::

   ┌───────────────┬────────┬────────┬──────────┬─────────────┬─────────────┬──────────────┐
   │ experiment_id ┆ dose_a ┆ dose_b ┆ response ┆ dose_a_resp ┆ dose_b_resp ┆ resp_imputed │
   │ ---           ┆ ---    ┆ ---    ┆ ---      ┆ ---         ┆ ---         ┆ ---          │
   │ u32           ┆ f64    ┆ f64    ┆ f64      ┆ f64         ┆ f64         ┆ f64          │
   ╞═══════════════╪════════╪════════╪══════════╪═════════════╪═════════════╪══════════════╡
   │ 1             ┆ 0.0    ┆ 0.0    ┆ 0.0      ┆ 0.04        ┆ 0.0         ┆ 0.0          │
   │ 1             ┆ 0.0    ┆ 0.0    ┆ 0.0      ┆ 0.04        ┆ 0.0         ┆ 0.0          │
   │ 1             ┆ 0.0    ┆ 0.0    ┆ 0.0      ┆ 0.04        ┆ 0.0         ┆ 0.0          │
   │ 1             ┆ 0.0    ┆ 0.0    ┆ 0.0      ┆ 0.04        ┆ 0.0         ┆ 0.0          │
   │ 1             ┆ 0.0    ┆ 0.0    ┆ 0.0      ┆ 0.04        ┆ 0.0         ┆ 0.0          │
   │ …             ┆ …      ┆ …      ┆ …        ┆ …           ┆ …           ┆ …            │
   │ 2             ┆ 1000.0 ┆ 3.16   ┆ null     ┆ 98.92       ┆ 23.23       ┆ 97.64        │
   │ 2             ┆ 1000.0 ┆ 56.23  ┆ null     ┆ 98.92       ┆ 85.1        ┆ 97.81        │
   │ 2             ┆ 1000.0 ┆ 1000.0 ┆ 99.99    ┆ 98.92       ┆ 99.08       ┆ 99.99        │
   │ 2             ┆ 1000.0 ┆ 1000.0 ┆ 99.99    ┆ 98.92       ┆ 99.08       ┆ 99.99        │
   │ 2             ┆ 1000.0 ┆ 1000.0 ┆ 99.99    ┆ 98.92       ┆ 99.08       ┆ 99.99        │
   └───────────────┴────────┴────────┴──────────┴─────────────┴─────────────┴──────────────┘

We note that the `IterativeImputer` response returns above 100 (a
telltale sign that something has gone awry), while this is not the case
with the `XGBoost` imputation. Due to its increased accuracy, it's
generally recommended to use the `XGBR` option when possible.

***************
 Factorization
***************

Now that we've pre-imputed a full matrix, we can feed this matrix to
matrix factorization algorithms. There are several algorithms available
to us - NMF, SVD, RPCA, and PMF - and we can pick and choose which ones
we want. For this example, let's use NMF and SVD. Xynergy makes this
relatively painless:

.. code:: python

   from xynergy.factor import matrix_factorize

   factored = matrix_factorize(
       imputed,
       method=["SVD", "NMF"],
       dose_cols=["dose_a", "dose_b"],
       response_col="resp_imputed",
       experiment_cols="experiment_id",
    )

    factored

.. code::

   ┌──────────┬─────────────┬─────────────┬──────────────┬───┬────────┬────────┬──────────────────┬──────────────────┐
   │ response ┆ dose_a_resp ┆ dose_b_resp ┆ resp_imputed ┆ … ┆ dose_a ┆ dose_b ┆ resp_imputed_NMF ┆ resp_imputed_SVD │
   │ ---      ┆ ---         ┆ ---         ┆ ---          ┆   ┆ ---    ┆ ---    ┆ ---              ┆ ---              │
   │ f64      ┆ f64         ┆ f64         ┆ f64          ┆   ┆ f64    ┆ f64    ┆ f64              ┆ f64              │
   ╞══════════╪═════════════╪═════════════╪══════════════╪═══╪════════╪════════╪══════════════════╪══════════════════╡
   │ 0.0      ┆ 0.04        ┆ 0.0         ┆ 0.0          ┆ … ┆ 0.0    ┆ 0.0    ┆ 0.01             ┆ -0.04            │
   │ 0.0      ┆ 0.04        ┆ 0.0         ┆ 0.0          ┆ … ┆ 0.0    ┆ 0.0    ┆ 0.01             ┆ -0.04            │
   │ 0.0      ┆ 0.04        ┆ 0.0         ┆ 0.0          ┆ … ┆ 0.0    ┆ 0.0    ┆ 0.01             ┆ -0.04            │
   │ 0.0      ┆ 0.04        ┆ 0.0         ┆ 0.0          ┆ … ┆ 0.0    ┆ 0.0    ┆ 0.01             ┆ -0.04            │
   │ 0.0      ┆ 0.04        ┆ 0.0         ┆ 0.0          ┆ … ┆ 0.0    ┆ 0.0    ┆ 0.01             ┆ -0.04            │
   │ …        ┆ …           ┆ …           ┆ …            ┆ … ┆ …      ┆ …      ┆ …                ┆ …                │
   │ null     ┆ 98.92       ┆ 23.23       ┆ 97.64        ┆ … ┆ 1000.0 ┆ 3.16   ┆ 97.07            ┆ 97.63            │
   │ null     ┆ 98.92       ┆ 85.1        ┆ 97.81        ┆ … ┆ 1000.0 ┆ 56.23  ┆ 98.88            ┆ 98.32            │
   │ 99.99    ┆ 98.92       ┆ 99.08       ┆ 99.99        ┆ … ┆ 1000.0 ┆ 1000.0 ┆ 98.88            ┆ 99.98            │
   │ 99.99    ┆ 98.92       ┆ 99.08       ┆ 99.99        ┆ … ┆ 1000.0 ┆ 1000.0 ┆ 98.88            ┆ 99.98            │
   │ 99.99    ┆ 98.92       ┆ 99.08       ┆ 99.99        ┆ … ┆ 1000.0 ┆ 1000.0 ┆ 98.88            ┆ 99.98            │
   └──────────┴─────────────┴─────────────┴──────────────┴───┴────────┴────────┴──────────────────┴──────────────────┘

.. NOTE::

   Like in the case of `pre_impute`, since we are using the default
   column names from `tidy`, the default argument names for `dose_cols`,
   `response_col`, and `experiment_cols` are the same as the one we
   provided, so this is equivalent:

   .. code:: python

      factored = matrix_factorize(imputed, method=["SVD", "NMF"])

Like `pre_impute`, `matrix_factorize` appends new columns to the input -
in this case, one for each method.

*****************
 Post-imputation
*****************

Finally, we can use the resultant columns from `matrix_factorize` to
predict a final response column:

.. code:: python

   final = post_impute(factored)
   final

.. NOTE::

   By default, this function uses any columns that start with
   `resp_imputed_` to impute the final response, but the columns used -
   or the prefix searched for - can be manually set. As is the pattern
   with Xynergy, the default outputs of the previous function are the
   default inputs of this function, so in this case we don't need to set
   anything.

.. code::

   ┌───────────┬─────────────┬─────────────┬──────────────┬───┬────────┬────────┬──────────────────┬──────────────────┐
   │ response  ┆ dose_a_resp ┆ dose_b_resp ┆ resp_imputed ┆ … ┆ dose_a ┆ dose_b ┆ resp_imputed_NMF ┆ resp_imputed_SVD │
   │ ---       ┆ ---         ┆ ---         ┆ ---          ┆   ┆ ---    ┆ ---    ┆ ---              ┆ ---              │
   │ f32       ┆ f64         ┆ f64         ┆ f64          ┆   ┆ f64    ┆ f64    ┆ f64              ┆ f64              │
   ╞═══════════╪═════════════╪═════════════╪══════════════╪═══╪════════╪════════╪══════════════════╪══════════════════╡
   │ 0.0       ┆ 0.11        ┆ 0.04        ┆ 0.0          ┆ … ┆ 0.0    ┆ 0.0    ┆ 0.0              ┆ -0.0             │
   │ 0.0       ┆ 0.11        ┆ 0.04        ┆ 0.0          ┆ … ┆ 0.0    ┆ 0.0    ┆ 0.0              ┆ -0.0             │
   │ 0.0       ┆ 0.11        ┆ 0.04        ┆ 0.0          ┆ … ┆ 0.0    ┆ 0.0    ┆ 0.0              ┆ -0.0             │
   │ 0.0       ┆ 0.11        ┆ 0.04        ┆ 0.0          ┆ … ┆ 0.0    ┆ 0.0    ┆ 0.0              ┆ -0.0             │
   │ 0.0       ┆ 0.11        ┆ 0.04        ┆ 0.0          ┆ … ┆ 0.0    ┆ 0.0    ┆ 0.0              ┆ -0.0             │
   │ …         ┆ …           ┆ …           ┆ …            ┆ … ┆ …      ┆ …      ┆ …                ┆ …                │
   │ 98.099998 ┆ 85.73       ┆ 98.94       ┆ 98.61        ┆ … ┆ 56.23  ┆ 1000.0 ┆ 99.02            ┆ 98.97            │
   │ 98.559998 ┆ 99.01       ┆ 0.1         ┆ 97.71        ┆ … ┆ 1000.0 ┆ 0.01   ┆ 101.08           ┆ 98.45            │
   │ 98.580002 ┆ 99.01       ┆ 1.71        ┆ 97.71        ┆ … ┆ 1000.0 ┆ 0.18   ┆ 100.37           ┆ 98.35            │
   │ 98.010002 ┆ 99.01       ┆ 23.3        ┆ 97.71        ┆ … ┆ 1000.0 ┆ 3.16   ┆ 94.99            ┆ 97.6             │
   │ 98.019997 ┆ 99.01       ┆ 84.16       ┆ 97.81        ┆ … ┆ 1000.0 ┆ 56.23  ┆ 98.73            ┆ 98.2             │
   └───────────┴─────────────┴─────────────┴──────────────┴───┴────────┴────────┴──────────────────┴──────────────────┘

Note that unlike most other functions in `Xynergy`, this does not add an
additional column, but modified a previously existing column (here
`response`). Whereas before our `response` column had `null` values in
it, those values have been imputed!

**********
 Accuracy
**********

Normally you won't have the luxury of knowing how close the responses
are to ground truth. In this particular case, I removed values from the
original dataset before imputing them, so we can compare the original
values to the imputed values to see how far off we were

.. code:: python

   # Collapse replicates to a single value for every dose-pair by taking the mean
   final_summary = final.group_by(
       ["experiment_id", "dose_a", "dose_b"],
   ).agg(
       pl.col("response").mean(),
   )

   # Join it with the original data that doesn't have any missing values
   with_og = og_data_summary.join(final_summary, on=["dose_a", "dose_b", "experiment_id"])

   # As a simple summary, we can take the root mean squared error:

   # "response" is our prediction, "resp" is the original data
   rmse = ((with_og["response"] - with_og["resp"]) ** 2).mean() ** 0.5
   rmse

.. code::

   1.1592251324998666

Pretty good! We can get a better picture - literally - of how predicted
and actual responses vary by using a built-in plotting helper function

**********
 Plotting
**********

.. code:: python

   from xynergy.plot import plot_response_landscape


   for i, group in with_og.group_by("experiment_id"):
       plot_response_landscape(
           df=group,
           dose_cols=["dose_a", "dose_b"],
           response_col="response",
           reference_col="resp",
           scheme="redblue",
           color_min=-20,
           color_mid=0,
           color_max=20,
       ).save(f"{i[0]}_vs-og.png", ppi=800)

`plot_reponse_landscape` plots the dose-response matrix, with doses as
the axis and response as the color. If a `reference_col` argument is
supplied, that column is subtracted from the `response_col`. So in this
case, we're plotting how far off our predicted values were from the
original values.

.. list-table:: Deviation of imputed response from true response

   -  -  .. figure:: ./_static/1_vs-og.png
            :width: 350px
      -  .. figure:: ./_static/2_vs-og.png
            :width: 350px

We can see from these plots that Xynergy did a pretty good job
estimating the original dataset - almost everything looks gray, implying
the difference between imputed response and actual response is near 0.

This function can also be useful for plotting differences from a given
synergy model. Let's talk about that now.

*********
 Synergy
*********

Simplistically, synergy is when a combination of drugs acts with greater
effect combined 'than expected' from their individual effects. Usually,
synergy models define what they would expect a no-synergy case (the null
or reference), and then see how much the observed deviations vary from
that.

Xynergy provides several synergy models - Bliss independence, highest
single agent (HSA), Loewe additivity, and zero interaction potency
(ZIP). Generally, the best way to calculate synergy is with the
`add_synergy` function:

.. code:: python

   from xynergy.synergy import add_synergy

   with_synergy = add_synergy(
      final, ["dose_a", "dose_b"], "response", "experiment_id", ["bliss", "zip"]
   )
   # Too many columns to show comfortably here - here are the important ones
   with_synergy[:, [8, 4, 9, 10, 13, 14]]

.. code::

   ┌───────────────┬───────────┬────────┬────────┬───────────┬─────────┐
   │ experiment_id ┆ response  ┆ dose_a ┆ dose_b ┆ bliss_syn ┆ zip_syn │
   │ ---           ┆ ---       ┆ ---    ┆ ---    ┆ ---       ┆ ---     │
   │ u32           ┆ f32       ┆ f64    ┆ f64    ┆ f32       ┆ f64     │
   ╞═══════════════╪═══════════╪════════╪════════╪═══════════╪═════════╡
   │ 1             ┆ 0.0       ┆ 0.0    ┆ 0.0    ┆ 0.0       ┆ -0.02   │
   │ 1             ┆ 0.0       ┆ 0.0    ┆ 0.0    ┆ 0.0       ┆ -0.02   │
   │ 1             ┆ 0.0       ┆ 0.0    ┆ 0.0    ┆ 0.0       ┆ -0.02   │
   │ 1             ┆ 0.0       ┆ 0.0    ┆ 0.0    ┆ 0.0       ┆ -0.02   │
   │ 1             ┆ 0.0       ┆ 0.0    ┆ 0.0    ┆ 0.0       ┆ -0.02   │
   │ …             ┆ …         ┆ …      ┆ …      ┆ …         ┆ …       │
   │ 1             ┆ 99.989998 ┆ 1000.0 ┆ 1000.0 ┆ -0.0      ┆ -0.01   │
   │ 1             ┆ 99.989998 ┆ 1000.0 ┆ 1000.0 ┆ 0.0       ┆ -0.01   │
   │ 2             ┆ 99.989998 ┆ 1000.0 ┆ 1000.0 ┆ -0.0      ┆ -0.0    │
   │ 2             ┆ 99.989998 ┆ 1000.0 ┆ 1000.0 ┆ -0.0      ┆ -0.0    │
   │ 2             ┆ 99.989998 ┆ 1000.0 ┆ 1000.0 ┆ 0.0       ┆ -0.0    │
   └───────────────┴───────────┴────────┴────────┴───────────┴─────────┘

We notice that a lot of these values are close to 0, implying that our
combinations exhibit very little synergy (or antagonism). It's much
easier to see these when plotting - let's plot the deviation from the
zero-interaction potency reference model (`zip_syn`):

.. code:: python

   for i, group in with_synergy.group_by("experiment_id"):
       plot_response_landscape(
          group, ["dose_a", "dose_b"],
          response_col="zip_syn",
          scheme="redblue",
          color_min=-20,
          color_mid=0,
          color_max=20,
       ).save(f"{i[0]}_zip.png", ppi=800)

.. list-table:: ZIP synergy landscape for each experiment

   -  -  .. figure:: ./_static/1_zip.png
            :width: 350px
      -  .. figure:: ./_static/2_zip.png
            :width: 350px

You'll note here that pretty much everything looks gray. There's a good
reason for this: I simulated these data to have 0 synergy under the
Bliss independence model, and ZIP and Bliss are very similar. A smarter
person would have created an example with an exciting synergy and
antagonism that could be revealed through this process, but I am not
that person.

.. DANGER::

   You may want to calculate just the reference model landscape for a
   given experiment. Xynergy *will* enable you to do this with
   `add_reference`, but prefer `add_synergy` if that's what you really
   want. For most *but not all* synergy models, subtracting the
   reference model from the observed responses will give you the same
   synergy score. However, ZIP does **not** do this and if you attempt
   to calculate synergy scores like this, you will get the wrong value.
